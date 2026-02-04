# Linear Label Catalog (VRAXION)

This folder contains a deterministic snapshot of the Linear label taxonomy in a
searchable form.

Why:
- Linear MCP exposes each label with `id`, `parentId`, `name`, etc.
- Group headers are not reliably searchable by name via MCP, so we reconstruct
  group/child structure from each label's `name` using `GROUP → CHILD`.
- We keep both raw and ASCII-normalized (`→` to `->`) columns to avoid Windows
  console encoding surprises while preserving exact Linear names.

## Files

- `labels_catalog_v1.csv` - grep-friendly catalog for humans
- `labels_catalog_v1.json` - machine-friendly catalog with IDs + warnings

## Regenerate the catalog

1) Dump labels via Linear MCP (team: VRAXION).

   Use the MCP tool `list_issue_labels`:
   - `team="VRAXION"`
   - `limit=250`
   - If `hasNextPage=true`, repeat with the returned `cursor`.

   Save the combined JSON to a file shaped like:
   ```json
   {"labels":[ ... ]}
   ```

2) Run the normalizer:

   ```powershell
   Set-Location -LiteralPath "S:\AI\work\VRAXION_DEV\Golden Draft"
   python tools\linear_labels_catalog.py --input <raw_labels.json> --out-dir docs\linear
   ```

The tool will emit:
- `docs/linear/labels_catalog_v1.csv`
- `docs/linear/labels_catalog_v1.json`

## Notes / Known quirks

- For grouped labels, MCP operations often work best when referencing the full
  label name (e.g., `EVIDENCE LEVEL → E1 PROBE`) rather than the group header
  alone.
- Some labels may intentionally have empty descriptions (e.g., unfinished FOCUS
  group). The catalog will warn on missing descriptions but will still write
  outputs.

