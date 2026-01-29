"""INSTNCT data loaders and synthetic stream builders.

Golden Draft module extracted from the legacy monolithic script.

Scope (intentionally narrow):
- Sequential MNIST loader (16x16 -> [256, 1] sequence)
- Synthetic dataset modes (VRX_SYNTH=1)
- Deterministic A/B synthetic pair loaders (lockout probe)

This module is intentionally behavior-conservative: other tooling expects
the same env variables and (roughly) the same dataset semantics as the
monolith.
"""

from __future__ import annotations

import json
import os
import random
import urllib.request
import zipfile
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset

from vraxion.instnct import infra
from vraxion.settings import load_settings

# Token ids used by some synthetic modes (assoc_mix BOS/domain/EOS injection).
BYTE_VOCAB = 256
BOS_ID = int(os.environ.get('VRX_BOS_ID', str(BYTE_VOCAB)))
EOS_ID = int(os.environ.get('VRX_EOS_ID', str(BYTE_VOCAB + 1)))
PAD_ID = int(os.environ.get('VRX_PAD_ID', str(BYTE_VOCAB + 2)))
SEP_ID = int(os.environ.get('VRX_SEP_ID', str(BYTE_VOCAB + 3)))
CODE_ID = int(os.environ.get('VRX_CODE_ID', str(BYTE_VOCAB + 4)))
TEXT_ID = int(os.environ.get('VRX_TEXT_ID', str(BYTE_VOCAB + 5)))
VISION_ID = int(os.environ.get('VRX_VISION_ID', str(BYTE_VOCAB + 6)))
AUDIO_ID = int(os.environ.get('VRX_AUDIO_ID', str(BYTE_VOCAB + 7)))

SYNTH_META: Dict[str, Any] = {}

def _download_zip(url: str, dest_dir: str, tag: str) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, f'{tag}.zip')
    if not os.path.exists(zip_path):
        infra.log(f'Downloading {tag}...')
        urllib.request.urlretrieve(url, zip_path)
    return zip_path


def _extract_zip(zip_path: str, dest_dir: str) -> None:
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(dest_dir)


@dataclass(frozen=True)
class _AudioItem:
    path: str
    label: int


class FileAudioDataset(Dataset):
    def __init__(
        self,
        items: list[_AudioItem],
        num_classes: int,
        *,
        sample_rate: int = 16000,
        max_len: int = 16000,
        n_mels: int = 64,
        max_frames: int = 100,
    ) -> None:
        self.items = list(items)
        self.num_classes = int(num_classes)
        self.sample_rate = int(sample_rate)
        self.max_len = int(max_len)
        self.max_frames = int(max_frames)

        try:
            import torchaudio  # type: ignore
        except Exception as exc:
            raise RuntimeError('torchaudio not available') from exc

        self._torchaudio = torchaudio
        self._melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=400,
            hop_length=160,
            n_mels=int(n_mels),
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        item = self.items[int(idx)]
        wav, sr = self._torchaudio.load(item.path)
        if wav.size(0) > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if int(sr) != self.sample_rate:
            wav = self._torchaudio.functional.resample(wav, int(sr), self.sample_rate)
        if wav.size(1) < self.max_len:
            pad = self.max_len - wav.size(1)
            wav = torch.nn.functional.pad(wav, (0, pad))
        else:
            wav = wav[:, : self.max_len]

        with torch.no_grad():
            mel = self._melspec(wav)
            mel = torch.log(mel + 1e-6)
        mel = mel.squeeze(0).transpose(0, 1)
        if mel.size(0) < self.max_frames:
            pad = self.max_frames - mel.size(0)
            mel = torch.nn.functional.pad(mel, (0, 0, 0, pad))
        else:
            mel = mel[: self.max_frames]
        return mel, int(item.label)


def get_fsdd_loader(*, batch_size: Optional[int] = None, max_samples: Optional[int] = None) -> Tuple[Any, int]:
    cfg = load_settings()
    bsz = int(batch_size if batch_size is not None else cfg.batch_size)
    maxs = int(max_samples if max_samples is not None else cfg.max_samples)

    root = os.path.join(cfg.data_dir, 'fsdd')
    if cfg.offline_only and not os.path.exists(root):
        raise RuntimeError('offline mode and fsdd not present')

    zip_url = 'https://github.com/Jakobovski/free-spoken-digit-dataset/archive/refs/heads/master.zip'
    zip_path = _download_zip(zip_url, root, 'fsdd')
    extract_root = os.path.join(root, 'free-spoken-digit-dataset-master')
    if not os.path.exists(extract_root):
        _extract_zip(zip_path, root)

    recordings = os.path.join(extract_root, 'recordings')
    items: list[_AudioItem] = []
    for fname in sorted(os.listdir(recordings)):
        if not fname.endswith('.wav'):
            continue
        label = int(fname.split('_')[0])
        items.append(_AudioItem(os.path.join(recordings, fname), label))
    if maxs:
        items = items[:maxs]

    dataset = FileAudioDataset(items, num_classes=10)
    loader = DataLoader(dataset, batch_size=bsz, shuffle=True, num_workers=0, pin_memory=True)
    return loader, 10

def get_seq_mnist_loader(
    train: bool = True,
    *,
    batch_size: Optional[int] = None,
    max_samples: Optional[int] = None,
) -> Tuple[Any, int, Callable[..., Any]]:
    cfg = load_settings()
    ROOT = str(cfg.root)
    DATA_DIR = str(cfg.data_dir)
    OFFLINE_ONLY = bool(cfg.offline_only)
    SEED = int(cfg.seed)
    MAX_SAMPLES = int(max_samples) if max_samples is not None else int(cfg.max_samples)
    BATCH_SIZE = int(batch_size) if batch_size is not None else int(cfg.batch_size)
    SYNTH_LEN = int(cfg.synth_len)
    SYNTH_SHUFFLE = bool(cfg.synth_shuffle)
    SYNTH_MODE = str(cfg.synth_mode).strip().lower()
    ASSOC_KEYS = int(cfg.assoc_keys)
    ASSOC_PAIRS = int(cfg.assoc_pairs)
    ASSOC_VAL_RANGE = int(cfg.assoc_val_range)
    HAND_MIN = int(cfg.hand_min)

    STAIRCASE_LENS_RAW = os.environ.get('VRX_STAIRCASE_LENS', '').strip()
    STAIRCASE_WEIGHTS_RAW = os.environ.get('VRX_STAIRCASE_WEIGHTS', '').strip()
    STAIRCASE_ENABLED = bool(STAIRCASE_LENS_RAW)
    STAIRCASE_ADAPT = os.environ.get('VRX_STAIRCASE_ADAPT', '0') == '1'
    STAIRCASE_ADAPT_EVERY = int(os.environ.get('VRX_STAIRCASE_ADAPT_EVERY', '2000'))
    STAIRCASE_MIN_BASE = float(os.environ.get('VRX_STAIRCASE_MIN_BASE', '0.60'))
    STAIRCASE_SHIFT = float(os.environ.get('VRX_STAIRCASE_SHIFT', '0.02'))
    STAIRCASE_STABLE_STD = float(os.environ.get('VRX_STAIRCASE_STABLE_STD', '0.02'))

    log = infra.log
    _parse_csv_ints = infra._parse_csv_ints
    _parse_csv_floats = infra._parse_csv_floats
    _default_staircase_weights = infra._default_staircase_weights
    StaircaseController = infra.StaircaseController
    StaircaseBatcher = infra.StaircaseBatcher

    SYNTH_META.clear()
    synth_env = os.environ.get("VRX_SYNTH", "0").strip()
    if synth_env == "1":
        synth_mode = SYNTH_MODE
        base_seq_len = max(1, int(SYNTH_LEN))
        SYNTH_META.update({"enabled": True, "mode": synth_mode, "synth_len": base_seq_len})
        n_samples = max(1, MAX_SAMPLES)
        staircase_lens = _parse_csv_ints(STAIRCASE_LENS_RAW) if STAIRCASE_ENABLED else None
        if STAIRCASE_ENABLED and not staircase_lens:
            log(f"[staircase] invalid lens '{STAIRCASE_LENS_RAW}', disabling")
        staircase_weights = None
        if staircase_lens:
            if STAIRCASE_WEIGHTS_RAW:
                staircase_weights = _parse_csv_floats(STAIRCASE_WEIGHTS_RAW)
                if not staircase_weights or len(staircase_weights) != len(staircase_lens):
                    log(f"[staircase] invalid weights '{STAIRCASE_WEIGHTS_RAW}', using defaults")
                    staircase_weights = None
            if staircase_weights is None:
                staircase_weights = _default_staircase_weights(staircase_lens)

        def _wrap_dataset(x_src: torch.Tensor, y_src: torch.Tensor):
            class _SynthDataset(torch.utils.data.Dataset):
                def __len__(self):
                    return x_src.size(0)

                def __getitem__(self, item):
                    return x_src[item], y_src[item]

            def collate(batch):
                xs, ys = zip(*batch)
                return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)

            return _SynthDataset(), collate
        if synth_mode == "boundary_stream":
            x_path = os.environ.get("VRX_BOUNDARY_X", os.path.join(ROOT, "data", "stm_boundary_x.npy"))
            y_path = os.environ.get("VRX_BOUNDARY_Y", os.path.join(ROOT, "data", "stm_boundary_y.npy"))
            x = torch.from_numpy(np.load(x_path)).float()
            y = torch.from_numpy(np.load(y_path)).long()
            seq_len = x.size(1)
            SYNTH_META.update({"boundary_stream": True, "synth_len": seq_len, "rows": int(x.size(0))})
            y_max = int(y.max().item()) if y.numel() else -1
            num_classes = max(256, y_max + 1)
            log(f"[synth] mode=boundary_stream rows={int(x.size(0))} len={seq_len} y_max={y_max} x={x_path} y={y_path}")
            ds, collate = _wrap_dataset(x, y)
            loader = DataLoader(
                ds,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=0,
                pin_memory=False,
                collate_fn=collate,
            )
            return loader, num_classes, collate
        if synth_mode == "assoc_clean":
            pairs = max(1, int(ASSOC_PAIRS))
            keys = max(2, int(ASSOC_KEYS))
            min_len = pairs * 2 + 1
            max_bumps = 5

            def _build_assoc(seq_len_local: int):
                x_local = torch.zeros((n_samples, seq_len_local, 1), dtype=torch.float32)
                y_local = torch.zeros((n_samples,), dtype=torch.long)
                max_start_local = seq_len_local - 3  # reserve last token for query
                for idx in range(n_samples):
                    used = set()
                    pair_specs = []
                    starts = list(range(0, max_start_local + 1))
                    random.shuffle(starts)
                    for cand in starts:
                        if cand in used or (cand + 1) in used:
                            continue
                        used.add(cand)
                        used.add(cand + 1)
                        key_id = random.randint(0, keys - 1)
                        val = random.randint(0, 1)
                        key_token = float(2 + key_id)
                        val_token = -1.0 if val == 0 else -2.0
                        x_local[idx, cand, 0] = key_token
                        x_local[idx, cand + 1, 0] = val_token
                        pair_specs.append((key_id, val, key_token))
                        if len(pair_specs) >= pairs:
                            break
                    if len(pair_specs) < pairs:
                        return None, None
                    _, q_val, q_token = random.choice(pair_specs)
                    x_local[idx, -1, 0] = q_token
                    y_local[idx] = q_val
                return x_local, y_local

            def _make_assoc_clean(seq_len_local: int):
                seq_len = seq_len_local
                bump_attempts = 0
                if seq_len < min_len:
                    log(f"[synth] assoc_clean bump len from {seq_len} to {min_len} (min_len)")
                    seq_len = min_len
                x = None
                y = None
                while bump_attempts <= max_bumps:
                    x, y = _build_assoc(seq_len)
                    if x is not None:
                        break
                    bump_attempts += 1
                    new_len = seq_len + max(2, pairs) * 2
                    log(f"[synth] assoc_clean bump len from {seq_len} to {new_len} (placement failed)")
                    seq_len = new_len
                if x is None:
                    raise RuntimeError("assoc_clean: failed to place non-overlapping pairs after bumps")
                return x, y, seq_len

            if staircase_lens:
                loaders = []
                lens_actual = []
                for seq_len_local in staircase_lens:
                    x, y, used_len = _make_assoc_clean(seq_len_local)
                    ds, collate = _wrap_dataset(x, y)
                    loader = DataLoader(
                        ds,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=0,
                        pin_memory=False,
                        collate_fn=collate,
                    )
                    loaders.append(loader)
                    lens_actual.append(used_len)
                num_classes = 2
                if lens_actual != staircase_lens:
                    log(f"[staircase] assoc_clean adjusted lens {staircase_lens} -> {lens_actual}")
                staircase = StaircaseController(
                    lens_actual,
                    staircase_weights,
                    STAIRCASE_MIN_BASE,
                    STAIRCASE_SHIFT,
                    STAIRCASE_STABLE_STD,
                    STAIRCASE_ADAPT_EVERY,
                )
                batcher = StaircaseBatcher(loaders, staircase_weights, SEED, staircase=staircase)
                SYNTH_META.update(
                    {
                        "assoc_keys": keys,
                        "assoc_pairs": pairs,
                        "synth_len": lens_actual[0] if lens_actual else base_seq_len,
                        "staircase_lens": lens_actual,
                        "staircase_weights": staircase.weights,
                    }
                )
                log(
                    f"[synth] mode=assoc_clean rows={int(n_samples)} keys={keys} pairs={pairs} "
                    f"lens={lens_actual} weights={staircase.weights}"
                )
                return batcher, num_classes, collate

            x, y, seq_len = _make_assoc_clean(base_seq_len)
            SYNTH_META.update({"assoc_keys": keys, "assoc_pairs": pairs, "synth_len": seq_len})
            num_classes = 2
            log(f"[synth] mode=assoc_clean rows={int(n_samples)} keys={keys} pairs={pairs} len={seq_len}")
            ds, collate = _wrap_dataset(x, y)
            loader = DataLoader(
                ds,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
                collate_fn=collate,
            )
            return loader, num_classes, collate
        elif synth_mode == "assoc_byte":
            pairs = max(1, int(ASSOC_PAIRS))
            keys = max(2, int(ASSOC_KEYS))
            val_range = max(2, int(ASSOC_VAL_RANGE))
            min_len = pairs * 2 + 1
            max_bumps = 5

            def _build_assoc_byte(seq_len_local: int):
                x_local = torch.zeros((n_samples, seq_len_local, 1), dtype=torch.float32)
                y_local = torch.zeros((n_samples,), dtype=torch.long)
                max_start_local = seq_len_local - 3  # reserve last token for query
                for idx in range(n_samples):
                    used = set()
                    pair_specs = []
                    starts = list(range(0, max_start_local + 1))
                    random.shuffle(starts)
                    for cand in starts:
                        if cand in used or (cand + 1) in used:
                            continue
                        used.add(cand)
                        used.add(cand + 1)
                        key_id = random.randint(0, keys - 1)
                        val = random.randint(0, val_range - 1)
                        key_token = float(2 + key_id)
                        val_token = -float(val + 1)  # keep value tokens distinct from keys/distractors
                        x_local[idx, cand, 0] = key_token
                        x_local[idx, cand + 1, 0] = val_token
                        pair_specs.append((key_id, val, key_token))
                        if len(pair_specs) >= pairs:
                            break
                    if len(pair_specs) < pairs:
                        return None, None
                    _, q_val, q_token = random.choice(pair_specs)
                    x_local[idx, -1, 0] = q_token
                    y_local[idx] = q_val
                return x_local, y_local

            def _make_assoc_byte(seq_len_local: int):
                seq_len = seq_len_local
                bump_attempts = 0
                if seq_len < min_len:
                    log(f"[synth] assoc_byte bump len from {seq_len} to {min_len} (min_len)")
                    seq_len = min_len
                x = None
                y = None
                while bump_attempts <= max_bumps:
                    x, y = _build_assoc_byte(seq_len)
                    if x is not None:
                        break
                    bump_attempts += 1
                    new_len = seq_len + max(2, pairs) * 2
                    log(f"[synth] assoc_byte bump len from {seq_len} to {new_len} (placement failed)")
                    seq_len = new_len
                if x is None:
                    raise RuntimeError("assoc_byte: failed to place non-overlapping pairs after bumps")
                return x, y, seq_len

            if staircase_lens:
                loaders = []
                lens_actual = []
                for seq_len_local in staircase_lens:
                    x, y, used_len = _make_assoc_byte(seq_len_local)
                    ds, collate = _wrap_dataset(x, y)
                    loader = DataLoader(
                        ds,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=0,
                        pin_memory=False,
                        collate_fn=collate,
                    )
                    loaders.append(loader)
                    lens_actual.append(used_len)
                num_classes = val_range
                if lens_actual != staircase_lens:
                    log(f"[staircase] assoc_byte adjusted lens {staircase_lens} -> {lens_actual}")
                staircase = StaircaseController(
                    lens_actual,
                    staircase_weights,
                    STAIRCASE_MIN_BASE,
                    STAIRCASE_SHIFT,
                    STAIRCASE_STABLE_STD,
                    STAIRCASE_ADAPT_EVERY,
                )
                batcher = StaircaseBatcher(loaders, staircase_weights, SEED, staircase=staircase)
                SYNTH_META.update(
                    {
                        "assoc_keys": keys,
                        "assoc_pairs": pairs,
                        "assoc_val_range": val_range,
                        "synth_len": lens_actual[0] if lens_actual else base_seq_len,
                        "staircase_lens": lens_actual,
                        "staircase_weights": staircase.weights,
                    }
                )
                log(
                    f"[synth] mode=assoc_byte rows={int(n_samples)} keys={keys} vals={val_range} "
                    f"pairs={pairs} lens={lens_actual} weights={staircase.weights}"
                )
                return batcher, num_classes, collate

            x, y, seq_len = _make_assoc_byte(base_seq_len)
            SYNTH_META.update(
                {"assoc_keys": keys, "assoc_pairs": pairs, "assoc_val_range": val_range, "synth_len": seq_len}
            )
            num_classes = val_range
            log(
                f"[synth] mode=assoc_byte rows={int(n_samples)} keys={keys} vals={val_range} "
                f"pairs={pairs} len={seq_len}"
            )
            ds, collate = _wrap_dataset(x, y)
            loader = DataLoader(
                ds,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
                collate_fn=collate,
            )
            return loader, num_classes, collate
        elif synth_mode == "assoc_mix":
            pairs = max(1, int(ASSOC_PAIRS))
            keys = max(2, int(ASSOC_KEYS))
            val_range = max(2, int(ASSOC_VAL_RANGE))
            min_len = pairs * 2 + 1
            max_bumps = 5
            n_samples_int = int(n_samples)
            n_clean = n_samples_int // 2
            n_byte = n_samples_int - n_clean

            def _build_assoc_clean(seq_len_local: int, n_samples_local: int):
                x_local = torch.zeros((n_samples_local, seq_len_local, 1), dtype=torch.float32)
                y_local = torch.zeros((n_samples_local,), dtype=torch.long)
                max_start_local = seq_len_local - 3  # reserve last token for query
                for idx in range(n_samples_local):
                    used = set()
                    pair_specs = []
                    starts = list(range(0, max_start_local + 1))
                    random.shuffle(starts)
                    for cand in starts:
                        if cand in used or (cand + 1) in used:
                            continue
                        used.add(cand)
                        used.add(cand + 1)
                        key_id = random.randint(0, keys - 1)
                        val = random.randint(0, 1)
                        key_token = float(2 + key_id)
                        val_token = -1.0 if val == 0 else -2.0
                        x_local[idx, cand, 0] = key_token
                        x_local[idx, cand + 1, 0] = val_token
                        pair_specs.append((key_id, val, key_token))
                        if len(pair_specs) >= pairs:
                            break
                    if len(pair_specs) < pairs:
                        return None, None
                    _, q_val, q_token = random.choice(pair_specs)
                    x_local[idx, -1, 0] = q_token
                    y_local[idx] = q_val
                return x_local, y_local

            def _build_assoc_byte(seq_len_local: int, n_samples_local: int):
                x_local = torch.zeros((n_samples_local, seq_len_local, 1), dtype=torch.float32)
                y_local = torch.zeros((n_samples_local,), dtype=torch.long)
                max_start_local = seq_len_local - 3  # reserve last token for query
                for idx in range(n_samples_local):
                    used = set()
                    pair_specs = []
                    starts = list(range(0, max_start_local + 1))
                    random.shuffle(starts)
                    for cand in starts:
                        if cand in used or (cand + 1) in used:
                            continue
                        used.add(cand)
                        used.add(cand + 1)
                        key_id = random.randint(0, keys - 1)
                        val = random.randint(0, val_range - 1)
                        key_token = float(2 + key_id)
                        val_token = -float(val + 1)  # keep value tokens distinct from keys/distractors
                        x_local[idx, cand, 0] = key_token
                        x_local[idx, cand + 1, 0] = val_token
                        pair_specs.append((key_id, val, key_token))
                        if len(pair_specs) >= pairs:
                            break
                    if len(pair_specs) < pairs:
                        return None, None
                    _, q_val, q_token = random.choice(pair_specs)
                    x_local[idx, -1, 0] = q_token
                    y_local[idx] = q_val
                return x_local, y_local

            mix_offset = float(os.environ.get("VRX_ASSOC_MIX_OFFSET", "100.0"))
            mix_clean_offset = float(os.environ.get("VRX_ASSOC_MIX_CLEAN_OFFSET", "0.0"))
            mix_domain_token = os.environ.get("VRX_ASSOC_MIX_DOMAIN_TOKEN") == "1"
            mix_clean_sentinel = float(os.environ.get("VRX_ASSOC_MIX_CLEAN_SENTINEL", str(CODE_ID)))
            mix_byte_sentinel = float(os.environ.get("VRX_ASSOC_MIX_BYTE_SENTINEL", str(TEXT_ID)))
            mix_bos_eos = os.environ.get("VRX_BOS_EOS_INJECT") == "1"

            def _prepend_domain_token(x_src, token):
                x_pad = torch.zeros(
                    (x_src.size(0), x_src.size(1) + 1, x_src.size(2)),
                    dtype=x_src.dtype,
                )
                x_pad[:, 1:, :] = x_src
                x_pad[:, 0, 0] = token
                return x_pad

            def _wrap_bos_domain_eos(x_src, token):
                if x_src.size(1) < 4:
                    return x_src
                x_pad = torch.full_like(x_src, float(PAD_ID))
                x_pad[:, 0, 0] = float(BOS_ID)
                x_pad[:, 1, 0] = float(token)
                max_copy = max(0, x_src.size(1) - 3)
                if max_copy > 0:
                    x_pad[:, 2:2 + max_copy, :] = x_src[:, :max_copy, :]
                x_pad[:, -1, 0] = float(EOS_ID)
                return x_pad

            def _make_assoc_mix(seq_len_local: int):
                seq_len = seq_len_local
                bump_attempts = 0
                if seq_len < min_len:
                    log(f"[synth] assoc_mix bump len from {seq_len} to {min_len} (min_len)")
                    seq_len = min_len
                x_clean = None
                y_clean = None
                x_byte = None
                y_byte = None
                while bump_attempts <= max_bumps:
                    x_clean, y_clean = _build_assoc_clean(seq_len, n_clean)
                    x_byte, y_byte = _build_assoc_byte(seq_len, n_byte)
                    if x_clean is not None and x_byte is not None:
                        break
                    bump_attempts += 1
                    new_len = seq_len + max(2, pairs) * 2
                    log(f"[synth] assoc_mix bump len from {seq_len} to {new_len} (placement failed)")
                    seq_len = new_len
                if x_clean is None or x_byte is None:
                    raise RuntimeError("assoc_mix: failed to place non-overlapping pairs after bumps")
                if mix_clean_offset:
                    mask = x_clean > 0
                    x_clean[mask] = x_clean[mask] + mix_clean_offset
                if mix_offset:
                    mask = x_byte > 0
                    x_byte[mask] = x_byte[mask] + mix_offset
                if mix_bos_eos:
                    x_clean = _wrap_bos_domain_eos(x_clean, mix_clean_sentinel if mix_domain_token else CODE_ID)
                    x_byte = _wrap_bos_domain_eos(x_byte, mix_byte_sentinel if mix_domain_token else TEXT_ID)
                elif mix_domain_token:
                    x_clean = _prepend_domain_token(x_clean, mix_clean_sentinel)
                    x_byte = _prepend_domain_token(x_byte, mix_byte_sentinel)
                    seq_len = seq_len + 1
                y_byte = y_byte + 2
                x = torch.cat([x_clean, x_byte], dim=0)
                y = torch.cat([y_clean, y_byte], dim=0)
                perm = torch.randperm(x.size(0))
                x = x[perm]
                y = y[perm]
                return x, y, seq_len

            if staircase_lens:
                loaders = []
                lens_actual = []
                for seq_len_local in staircase_lens:
                    x, y, used_len = _make_assoc_mix(seq_len_local)
                    ds, collate = _wrap_dataset(x, y)
                    loader = DataLoader(
                        ds,
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=0,
                        pin_memory=False,
                        collate_fn=collate,
                    )
                    loaders.append(loader)
                    lens_actual.append(used_len)
                num_classes = val_range + 2
                if lens_actual != staircase_lens:
                    log(f"[staircase] assoc_mix adjusted lens {staircase_lens} -> {lens_actual}")
                staircase = StaircaseController(
                    lens_actual,
                    staircase_weights,
                    STAIRCASE_MIN_BASE,
                    STAIRCASE_SHIFT,
                    STAIRCASE_STABLE_STD,
                    STAIRCASE_ADAPT_EVERY,
                )
                batcher = StaircaseBatcher(loaders, staircase_weights, SEED, staircase=staircase)
                SYNTH_META.update(
                    {
                        "assoc_keys": keys,
                        "assoc_pairs": pairs,
                        "assoc_val_range": val_range,
                        "synth_len": lens_actual[0] if lens_actual else base_seq_len,
                        "assoc_mix_clean": int(n_clean),
                        "assoc_mix_byte": int(n_byte),
                        "assoc_mix_offset": mix_offset,
                        "assoc_mix_clean_offset": mix_clean_offset,
                        "assoc_mix_domain_token": int(mix_domain_token),
                        "assoc_mix_clean_sentinel": mix_clean_sentinel,
                        "assoc_mix_byte_sentinel": mix_byte_sentinel,
                        "staircase_lens": lens_actual,
                        "staircase_weights": staircase.weights,
                    }
                )
                log(
                    f"[synth] mode=assoc_mix rows={int(n_samples_int)} clean={n_clean} byte={n_byte} "
                    f"keys={keys} vals={val_range} pairs={pairs} lens={lens_actual} "
                    f"offsets=clean:{mix_clean_offset:g} byte:{mix_offset:g} "
                    f"sentinel={int(mix_domain_token)} weights={staircase.weights}"
                )
                return batcher, num_classes, collate

            x, y, seq_len = _make_assoc_mix(base_seq_len)
            SYNTH_META.update(
                {
                    "assoc_keys": keys,
                    "assoc_pairs": pairs,
                    "assoc_val_range": val_range,
                    "synth_len": seq_len,
                    "assoc_mix_clean": int(n_clean),
                    "assoc_mix_byte": int(n_byte),
                    "assoc_mix_offset": mix_offset,
                    "assoc_mix_clean_offset": mix_clean_offset,
                    "assoc_mix_domain_token": int(mix_domain_token),
                    "assoc_mix_clean_sentinel": mix_clean_sentinel,
                    "assoc_mix_byte_sentinel": mix_byte_sentinel,
                }
            )
            num_classes = val_range + 2
            log(
                f"[synth] mode=assoc_mix rows={int(n_samples_int)} clean={n_clean} byte={n_byte} "
                f"keys={keys} vals={val_range} pairs={pairs} len={seq_len} "
                f"offsets=clean:{mix_clean_offset:g} byte:{mix_offset:g} "
                f"sentinel={int(mix_domain_token)}"
            )
            ds, collate = _wrap_dataset(x, y)
            loader = DataLoader(
                ds,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
                collate_fn=collate,
            )
            return loader, num_classes, collate
        elif synth_mode == "hand_kv":
            hand_path = os.environ.get("VRX_HAND_PATH", os.path.join(DATA_DIR, "hand_kv.jsonl"))
            pad_len = int(os.environ.get("VRX_HAND_PAD_LEN", "0"))
            rows = []
            with open(hand_path, "r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    rows.append(json.loads(line))
            if MAX_SAMPLES and MAX_SAMPLES < len(rows):
                rows = rows[:MAX_SAMPLES]
            if len(rows) < HAND_MIN:
                raise RuntimeError(
                    f"hand_kv dataset too small: {len(rows)} rows < HAND_MIN={HAND_MIN} ({hand_path})"
                )
            SYNTH_META.update({"hand_path": hand_path, "rows": len(rows), "pad_len": pad_len})
            log(f"[synth] mode=hand_kv rows={len(rows)} pad_len={pad_len} path={hand_path}")

            xs = []
            ys = []
            for row in rows:
                seq = row.get("x", [])
                label = row.get("y", 0)
                if pad_len > 0:
                    if len(seq) < pad_len:
                        seq = seq + [0] * (pad_len - len(seq))
                    else:
                        seq = seq[:pad_len]
                xs.append(torch.tensor(seq, dtype=torch.float32).view(-1, 1))
                ys.append(int(label))

            class _ListSynth(torch.utils.data.Dataset):
                def __len__(self):
                    return len(xs)

                def __getitem__(self, idx):
                    return xs[idx], ys[idx]

            ds = _ListSynth()

            def collate(batch):
                xs_b, ys_b = zip(*batch)
                return torch.stack(xs_b, dim=0), torch.tensor(ys_b, dtype=torch.long)

            loader = DataLoader(
                ds,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
                collate_fn=collate,
            )
            num_classes = max(2, max(ys) + 1 if ys else 2)
            return loader, num_classes, collate
        else:
            x = torch.randint(0, 2, (n_samples, base_seq_len, 1), dtype=torch.float32)
            if synth_mode == "markov0":
                y = x[:, -1, 0].to(torch.long)
            elif synth_mode == "markov0_flip":
                y = (1 - x[:, -1, 0]).to(torch.long)
            elif synth_mode == "const0":
                y = torch.zeros((n_samples,), dtype=torch.long)
            else:
                y = torch.randint(0, 2, (n_samples,), dtype=torch.long)
        SYNTH_META.update({"rows": int(n_samples)})
        log(f"[synth] mode={synth_mode} rows={int(n_samples)}")

        class _Synth(torch.utils.data.Dataset):
            def __len__(self):
                return n_samples

            def __getitem__(self, idx):
                return x[idx], y[idx]

        ds = _Synth()

        def collate(batch):
            xs, ys = zip(*batch)
            return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)

        loader = DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=SYNTH_SHUFFLE,
            num_workers=0,
            pin_memory=False,
            collate_fn=collate,
        )
        return loader, 2, collate

    try:
        import torchvision.transforms as T
        from torchvision.datasets import MNIST
    except Exception as exc:
        raise RuntimeError("torchvision is required for MNIST mode") from exc

    transform = T.Compose([T.Resize((16, 16)), T.ToTensor()])
    ds = MNIST(os.path.join(DATA_DIR, "mnist_seq"), train=bool(train), download=not OFFLINE_ONLY, transform=transform)
    if MAX_SAMPLES and MAX_SAMPLES < len(ds):
        ds = Subset(ds, list(range(MAX_SAMPLES)))

    def collate(batch):
        xs, ys = zip(*batch)
        x = torch.stack(xs, dim=0)  # [B,1,16,16]
        x = x.view(x.size(0), -1, 1)  # [B,256,1]
        y = torch.tensor(ys, dtype=torch.long)
        return x, y

    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate,
    )
    return loader, 10, collate



def build_synth_pair_loaders(*, batch_size: Optional[int] = None) -> Tuple[Any, Any, Callable[..., Any]]:
    cfg = load_settings()
    SEED = int(cfg.seed)
    MAX_SAMPLES = int(cfg.max_samples)
    SYNTH_LEN = int(cfg.synth_len)
    SYNTH_SHUFFLE = bool(cfg.synth_shuffle)
    BATCH_SIZE = int(batch_size) if batch_size is not None else int(cfg.batch_size)

    n_samples = max(1, MAX_SAMPLES)
    seq_len = max(1, SYNTH_LEN)
    g = torch.Generator()
    g.manual_seed(SEED)
    x = torch.randint(0, 2, (n_samples, seq_len, 1), dtype=torch.float32, generator=g)
    y_a = x[:, -1, 0].to(torch.long)
    y_b = (1 - x[:, -1, 0]).to(torch.long)

    class _FixedSynth(torch.utils.data.Dataset):
        def __init__(self, xs, ys):
            self.xs = xs
            self.ys = ys

        def __len__(self):
            return self.xs.size(0)

        def __getitem__(self, idx):
            return self.xs[idx], self.ys[idx]

    ds_a = _FixedSynth(x, y_a)
    ds_b = _FixedSynth(x, y_b)

    def collate(batch):
        xs, ys = zip(*batch)
        return torch.stack(xs, dim=0), torch.tensor(ys, dtype=torch.long)

    loader_a = DataLoader(
        ds_a,
        batch_size=BATCH_SIZE,
        shuffle=SYNTH_SHUFFLE,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate,
    )
    loader_b = DataLoader(
        ds_b,
        batch_size=BATCH_SIZE,
        shuffle=SYNTH_SHUFFLE,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate,
    )
    return loader_a, loader_b, collate




__all__ = [
    'SYNTH_META',
    'FileAudioDataset',
    'get_fsdd_loader',
    'get_seq_mnist_loader',
    'build_synth_pair_loaders',
]

