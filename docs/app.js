/* APOCALYPSE EXPRESS // CORE_LOGIC */

(function () {
  const $ = (sel) => document.querySelector(sel);
  const $$ = (sel) => document.querySelectorAll(sel);

  const REPO_URL = 'https://github.com/VRAXION/VRAXION';

  // --- RESPONSIVE CRT SCALE ---
  // The UI is authored at a fixed 640x360 "signal" resolution.
  // We scale it to fit any viewport, while preserving the intended look.
  const setCrtScale = () => {
    const BASE_W = 640;
    const BASE_H = 360;
    const MAX_SCALE = 2;
    const MARGIN = 24;

    const sx = (window.innerWidth - MARGIN * 2) / BASE_W;
    const sy = (window.innerHeight - MARGIN * 2) / BASE_H;
    const s = Math.max(0.6, Math.min(MAX_SCALE, sx, sy));
    document.documentElement.style.setProperty('--crt-scale', s.toFixed(3));
  };

  window.addEventListener('resize', setCrtScale);
  setCrtScale();

  const FSM = {
    state: 'boot',
    to(newState) {
      $$('.state').forEach(el => el.classList.remove('active'));
      document.getElementById(`state-${newState}`).classList.add('active');
      this.state = newState;
      document.body.classList.toggle('mode-reader', newState === 'reader');
    }
  };

  // --- BOOT SEQUENCE (CRYSIS TERMINAL + SKIP + CLEANUP) ---
  const bootSequence = async () => {
    // Clear screen first
    const terminal = document.getElementById('boot-terminal');
    if (terminal) terminal.innerHTML = '';

    // SKIP LOGIC
    let skipped = false;

    // Handler must be named to be removed
    const skipBootHandler = () => {
      // If we are not in boot state, ignore
      if (skipped || FSM.state !== 'boot') return;

      // SAFETY BUFFER: Ignore inputs for first 100ms
      // This prevents the "Reboot Click" from instantly skipping the boot
      if (Date.now() - bootStartTime < 100) return;

      skipped = true;
    };

    const bootStartTime = Date.now();

    // Add Listeners
    document.addEventListener('keydown', skipBootHandler);
    document.addEventListener('click', skipBootHandler);

    // Cleanup Helper
    const cleanup = () => {
      document.removeEventListener('keydown', skipBootHandler);
      document.removeEventListener('click', skipBootHandler);
    };

    try {
      const check = () => skipped;

      // Async Sequence
      await runTerminalLine("BIOS: VRAXION_AXIS [INIT]", 15, 60, check);
      if (!skipped) await runTerminalLine("KERNEL: INSTNCT v1.0.0 [LOAD]", 25, 40, check);
      if (!skipped) await runTerminalLine("PTR_DYNAMICS: ∇θ [SYNC]", 20, 80, check);
      if (!skipped) await runTerminalLine("SYSTEM: ONLINE [READY]", 30, 30, check);

      // End of sequence (or skipped)
      cleanup();
      setTimeout(() => FSM.to('menu'), skipped ? 0 : 600); // Instant if skipped

    } catch (e) {
      console.log("Boot error", e);
      cleanup();
      FSM.to('menu');
    }
  };

  // Line Generator: "TEXT [SPINNER] ........"
  const runTerminalLine = (textStr, maxDots, speed, isSkipped) => {
    return new Promise(resolve => {
      if (isSkipped && isSkipped()) { resolve(); return; }

      const parent = document.getElementById('boot-terminal');
      if (!parent) { resolve(); return; } // Safety

      const line = document.createElement('div');
      line.className = 'boot-line';
      line.innerHTML = `<span class="text">${textStr}</span> <span class="spinner-zone">/</span> <span class="dot-stream"></span>`;
      parent.appendChild(line);

      const spinnerEl = line.querySelector('.spinner-zone');
      const dotsEl = line.querySelector('.dot-stream');
      const chars = ['/', '-', '\\', '|'];
      let frame = 0;
      let dotCount = 0;

      const timer = setInterval(() => {
        if (isSkipped && isSkipped()) {
          clearInterval(timer);
          resolve();
          return;
        }

        frame++;
        spinnerEl.innerText = chars[frame % 4];
        dotCount++;
        dotsEl.innerText = ".".repeat(dotCount);

        if (dotCount >= maxDots) {
          clearInterval(timer);
          spinnerEl.innerText = "";
          spinnerEl.innerHTML = "<span style='color:#0f0'>[OK]</span>";
          resolve();
        }
      }, speed);
    });
  };

  // --- MENU LOGIC ---
  const menuItems = Array.from($$('.menu-item'));
  let menuIndex = 0;

  // Menu is a 2-column grid. Arrow keys move selection; Enter opens.
  const activateMenuIndex = (idx) => {
    if (!menuItems.length) return null;

    menuIndex = Math.max(0, Math.min(menuItems.length - 1, idx));
    const targetId = menuItems[menuIndex].dataset.target;

    menuItems.forEach((el, i) => el.classList.toggle('active', i === menuIndex));
    return targetId;
  };

  const setActiveMenu = (targetId) => {
    const idx = menuItems.findIndex(el => el.dataset.target === targetId);
    if (idx >= 0) activateMenuIndex(idx);
  };

  // Default selection
  activateMenuIndex(0);

  // Click to open
  menuItems.forEach((item, idx) => {
    item.addEventListener('click', () => {
      activateMenuIndex(idx);
      const target = item.dataset.target;
      loadReader(target);
      FSM.to('reader');
    });
  });

  const loadReader = (id) => {
    const src = document.getElementById(`data-${id}`);
    const dest = document.getElementById('doc-content');
    const titleEl = document.getElementById('doc-title');

    if (src && dest) {
      dest.innerHTML = src.innerHTML;
      dest.scrollTop = 0;

      if (titleEl) {
        const h1 = src.querySelector('h1');
        titleEl.textContent = h1
          ? h1.textContent.replace(/\s+/g, ' ').trim()
          : 'DOCUMENT_VIEWER';
      }
    }
  };

  // --- INPUT / HOTKEYS ---
  document.addEventListener('keydown', (e) => {
    const key = e.key.toLowerCase();

    // ESC to return from Reader
    if (key === 'escape' && FSM.state === 'reader') {
      FSM.to('menu');
    }

    // MENU HOTKEYS
    if (FSM.state === 'menu') {

      // Arrow navigation (2-column grid) + Enter to open
      if (key.startsWith('arrow') || key === 'enter') {
        e.preventDefault();

        const col = menuIndex % 2;

        if (key === 'arrowright' && col === 0) activateMenuIndex(menuIndex + 1);
        else if (key === 'arrowleft' && col === 1) activateMenuIndex(menuIndex - 1);
        else if (key === 'arrowdown') activateMenuIndex(menuIndex + 2);
        else if (key === 'arrowup') activateMenuIndex(menuIndex - 2);
        else if (key === 'enter') {
          const target = menuItems[menuIndex]?.dataset?.target;
          if (target) {
            loadReader(target);
            FSM.to('reader');
          }
        }

        return;
      }
      // G to open the GitHub repo
      if (key === 'g') {
        window.open(REPO_URL, '_blank', 'noopener');
        return;
      }

      const map = {
        'm': 'manifest',
        'i': 'instnct',
        'e': 'evidence',
        'h': 'hypothesis',
        'p': 'partners',
        'l': 'license'
      };

      if (map[key]) {
        const target = map[key];
        setActiveMenu(target);
        loadReader(target);
        FSM.to('reader');
      }

      // R to Reboot
      if (key === 'r') {
        FSM.to('boot');
        bootSequence();
      }
    }
  });

  // --- REBOOT BUTTON ---
  // --- REPO BUTTON ---
  const repoBtn = document.getElementById('btn-repo');
  if (repoBtn) {
    repoBtn.addEventListener('click', (e) => {
      e.stopPropagation();
      window.open(REPO_URL, '_blank', 'noopener');
    });
  }

  
  const rebootBtn = document.getElementById('btn-reboot');
  if (rebootBtn) {
    rebootBtn.addEventListener('click', (e) => {
      // PREVENT BUBBLING so it doesn't trigger "Skip" immediately
      e.stopPropagation();

      if (FSM.state === 'menu') {
        FSM.to('boot');
        bootSequence();
      }
    });
  }

  // --- RETURN BUTTON ---
  const returnBtn = document.getElementById('btn-return');
  if (returnBtn) {
    returnBtn.addEventListener('click', () => {
      if (FSM.state === 'reader') {
        FSM.to('menu');
      }
    });
  }

  // START
  bootSequence();

  // --- NEURO-INTERACTION (PARALLAX & SPARKS) ---
  const initBackgroundFX = () => {
    const bg = $('.phosphene-void');
    if (!bg) return;

    // Keep the background slightly "overscanned" so parallax never reveals empty edges.
    // (We translate a few px, then scale up to maintain full coverage.)
    const OVERSCAN_SCALE = 1.08;
    const MAX_SHIFT_PX = 28;

    let targetX = 0, targetY = 0;
    let currentX = 0, currentY = 0;

    // 1. PARALLAX & SPAWNER
    document.addEventListener('mousemove', (e) => {
      // Calc Parallax Target
      const x = (window.innerWidth - e.clientX * 2) / 100;
      const y = (window.innerHeight - e.clientY * 2) / 100;

      // Clamp so we never shift far enough to expose the body background.
      targetX = Math.max(-MAX_SHIFT_PX, Math.min(MAX_SHIFT_PX, x));
      targetY = Math.max(-MAX_SHIFT_PX, Math.min(MAX_SHIFT_PX, y));

      // Spawn Spark on fast movement (optional) or just randomize based on activity
      if (Math.random() < 0.05) spawnSpark(e.clientX, e.clientY);
    });

    document.addEventListener('click', (e) => {
      // Spawn Cluster
      for (let i = 0; i < 5; i++) {
        spawnSpark(e.clientX, e.clientY, true);
      }
    });

    // Smooth Parallax Loop
    const animateBg = () => {
      currentX += (targetX - currentX) * 0.05;
      currentY += (targetY - currentY) * 0.05;
      bg.style.transform = `translate(${currentX}px, ${currentY}px) scale(${OVERSCAN_SCALE})`;
      requestAnimationFrame(animateBg);
    };
    animateBg();
  };

  const spawnSpark = (x, y, burst = false) => {
    const spark = document.createElement('div');
    spark.className = 'neural-spark';

    // Randomize Brand Color
    const colors = [
      'rgba(188, 19, 254, 0.6)', // Purple
      'rgba(0, 240, 255, 0.6)',  // Blue
      'rgba(224, 240, 255, 0.6)' // White
    ];
    const color = colors[Math.floor(Math.random() * colors.length)];

    // Physics
    const size = 20 + Math.random() * 40;
    const offsetX = (Math.random() - 0.5) * (burst ? 100 : 20);
    const offsetY = (Math.random() - 0.5) * (burst ? 100 : 20);

    spark.style.width = `${size}px`;
    spark.style.height = `${size}px`;
    spark.style.background = `radial-gradient(circle, ${color} 0%, transparent 70%)`;
    spark.style.left = `${x + offsetX}px`;
    spark.style.top = `${y + offsetY}px`;

    // Inject
    document.querySelector('.phosphene-void').appendChild(spark);

    // Cleanup
    setTimeout(() => spark.remove(), 2000);
  };

  const initSynapticDust = () => {
    const bg = document.querySelector('.phosphene-void');
    if (!bg) return;

    setInterval(() => {
      const dust = document.createElement('div');
      dust.className = 'synapse-dust';

      // Random Start Pos
      const x = Math.random() * window.innerWidth;
      const y = Math.random() * window.innerHeight;

      // Random Drift Vector
      const tx = (Math.random() - 0.5) * 200 + 'px'; // +/- 100px drift
      const ty = (Math.random() - 0.5) * 200 + 'px';

      // Random Properties
      const size = (2 + Math.random() * 2) + 'px'; // 2px - 4px (Guaranteed Visible)
      const op = 0.5 + Math.random() * 0.5;        // 0.5 - 1.0 (Guaranteed Visible)

      dust.style.left = x + 'px';
      dust.style.top = y + 'px';
      dust.style.width = size;
      dust.style.height = size;
      dust.style.backgroundColor = 'rgba(224, 240, 255, 0.9)'; // Cyan Tint
      dust.style.setProperty('--tx', tx);
      dust.style.setProperty('--ty', ty);
      dust.style.setProperty('--op', op);

      bg.appendChild(dust);

      // Cleanup (Animation Duration is 15s)
      setTimeout(() => dust.remove(), 15000);

    }, 300); // Spawn rate
  };

  // Init FX
  initBackgroundFX();
  initSynapticDust();

})();
