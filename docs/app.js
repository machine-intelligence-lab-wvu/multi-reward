/* Animated telemetry charts — vanilla SVG/canvas, no external libraries. */

const REDUCED = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
const NS = 'http://www.w3.org/2000/svg';
const el = (n, a = {}) => { const e = document.createElementNS(NS, n); for (const k in a) e.setAttribute(k, a[k]); return e; };
const COL = { grpo: 'var(--sig-sup)', intuitor: 'var(--sig-fail)', ours: 'var(--sig-ours)', plain: 'var(--sig-sup)', aug: 'var(--sig-ours)' };

/* ---------------- line chart ---------------- */
function lineChart(host, cfg) {
  const W = 560, H = 300, P = { t: 18, r: 16, b: 38, l: 44 };
  const xs = cfg.steps, yMin = cfg.yMin, yMax = cfg.yMax;
  const px = s => P.l + (s - xs[0]) / (xs[xs.length - 1] - xs[0]) * (W - P.l - P.r);
  const py = v => H - P.b - (v - yMin) / (yMax - yMin) * (H - P.t - P.b);

  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, class: 'chart', role: 'img', 'aria-label': cfg.aria || 'trajectory chart' });

  // horizontal grid + y ticks
  const ticks = cfg.yTicks || 5;
  for (let i = 0; i <= ticks; i++) {
    const v = yMin + (yMax - yMin) * i / ticks, y = py(v);
    svg.appendChild(el('line', { x1: P.l, x2: W - P.r, y1: y, y2: y, class: 'grid' }));
    const t = el('text', { x: P.l - 8, y: y + 3.5, class: 'tick ty' }); t.textContent = Math.round(v);
    svg.appendChild(t);
  }
  // x ticks
  xs.forEach((s, i) => {
    if (xs.length > 8 && i % 2) return;
    const t = el('text', { x: px(s), y: H - P.b + 16, class: 'tick tx' }); t.textContent = s;
    svg.appendChild(t);
  });
  const xl = el('text', { x: (P.l + W - P.r) / 2, y: H - 4, class: 'axlab' }); xl.textContent = 'training step';
  svg.appendChild(xl);
  const yl = el('text', { x: 12, y: (P.t + H - P.b) / 2, class: 'axlab', transform: `rotate(-90 12 ${(P.t + H - P.b) / 2})` });
  yl.textContent = cfg.yLabel || 'pass@1 (%)'; svg.appendChild(yl);

  // base-model reference
  if (cfg.base != null) {
    svg.appendChild(el('line', { x1: P.l, x2: W - P.r, y1: py(cfg.base), y2: py(cfg.base), class: 'baseline' }));
    const b = el('text', { x: W - P.r - 2, y: py(cfg.base) - 6, class: 'baselab', 'text-anchor': 'end' });
    b.textContent = `base model ${cfg.base}`; svg.appendChild(b);
  }

  const paths = [];
  cfg.series.forEach(s => {
    const d = s.v.map((v, i) => `${i ? 'L' : 'M'}${px(xs[i]).toFixed(1)},${py(v).toFixed(1)}`).join(' ');
    const p = el('path', { d, class: 'ln', stroke: COL[s.k], 'stroke-width': s.k === 'ours' || s.k === 'aug' ? 2.8 : 2 });
    if (s.dash) p.setAttribute('stroke-dasharray', '5 4');
    svg.appendChild(p);
    const dot = el('circle', { r: 4.2, fill: COL[s.k], class: 'head' });
    svg.appendChild(dot);
    paths.push({ p, dot, s });
  });

  // collapse marker
  if (cfg.mark) {
    const g = el('g', { class: 'markg' });
    g.appendChild(el('line', { x1: px(cfg.mark.step), x2: px(cfg.mark.step), y1: P.t, y2: H - P.b, class: 'markline' }));
    const tx = el('text', { x: px(cfg.mark.step) + 5, y: P.t + 12, class: 'marktx' }); tx.textContent = cfg.mark.label;
    g.appendChild(tx); svg.appendChild(g);
  }

  host.appendChild(svg);

  // animate
  const run = () => {
    paths.forEach(({ p, dot, s }) => {
      const L = p.getTotalLength();
      p.style.transition = 'none';
      p.style.strokeDasharray = s.dash ? '5 4' : L; p.style.strokeDashoffset = s.dash ? 0 : L;
      dot.style.opacity = 0;
      if (REDUCED) { p.style.strokeDashoffset = 0; const e = p.getPointAtLength(L); dot.setAttribute('cx', e.x); dot.setAttribute('cy', e.y); dot.style.opacity = 1; return; }
      requestAnimationFrame(() => {
        const dur = 1700, t0 = performance.now();
        const tick = now => {
          const k = Math.min(1, (now - t0) / dur), e = 1 - Math.pow(1 - k, 3);
          if (!s.dash) p.style.strokeDashoffset = L * (1 - e);
          const pt = p.getPointAtLength(L * e);
          dot.setAttribute('cx', pt.x); dot.setAttribute('cy', pt.y); dot.style.opacity = 1;
          if (k < 1) requestAnimationFrame(tick);
        };
        requestAnimationFrame(tick);
      });
    });
    const mk = svg.querySelector('.markg');
    if (mk) { mk.style.opacity = 0; setTimeout(() => { mk.style.opacity = 1; }, 1200); }
  };
  return { run, svg };
}

/* ---------------- build charts ---------------- */
function buildTrajectory() {
  const host = document.getElementById('traj-chart');
  const bench = () => document.querySelector('#traj-tabs .on').dataset.b;
  const draw = () => {
    host.innerHTML = '';
    const b = bench(), d = DATA.qwen3[b];
    const c = lineChart(host, {
      steps: DATA.qwen3.steps,
      yMin: b === 'gsm8k' ? 40 : 40, yMax: b === 'gsm8k' ? 90 : 70,
      base: b === 'gsm8k' ? DATA.qwen3.base.gsm8k : DATA.qwen3.base.math500,
      aria: `Pass@1 over training on ${b} for Qwen3-1.7B-Base`,
      series: [
        { k: 'grpo', v: d.grpo, dash: true },
        { k: 'intuitor', v: d.intuitor },
        { k: 'ours', v: d.ours }
      ]
    });
    c.run(); return c;
  };
  let cur = draw();
  document.querySelectorAll('#traj-tabs button').forEach(btn => btn.addEventListener('click', () => {
    document.querySelectorAll('#traj-tabs button').forEach(x => x.classList.remove('on'));
    btn.classList.add('on'); cur = draw();
  }));
  document.getElementById('traj-replay').addEventListener('click', () => { cur = draw(); });
  return () => { cur = draw(); };
}

function buildKlcov() {
  const host = document.getElementById('klcov-chart');
  const bench = () => document.querySelector('#klcov-tabs .on').dataset.b;
  const draw = () => {
    host.innerHTML = '';
    const b = bench(), d = DATA.klcov[b];
    const c = lineChart(host, {
      steps: DATA.klcov.steps,
      yMin: b === 'gsm8k' ? 55 : 42, yMax: b === 'gsm8k' ? 80 : 62,
      aria: `GRPO-GT with and without KL-Cov on ${b}`,
      series: [{ k: 'plain', v: d.plain, dash: true }, { k: 'aug', v: d.aug }]
    });
    c.run(); return c;
  };
  let cur = draw();
  document.querySelectorAll('#klcov-tabs button').forEach(btn => btn.addEventListener('click', () => {
    document.querySelectorAll('#klcov-tabs button').forEach(x => x.classList.remove('on'));
    btn.classList.add('on'); cur = draw();
  }));
  return () => { cur = draw(); };
}

/* ---------------- retention bars ---------------- */
function buildRetention() {
  const host = document.getElementById('ret-rows');
  DATA.retention.forEach(r => {
    const row = document.createElement('div'); row.className = 'retrow';
    row.innerHTML = `<div class="retname">${r.back}<span>${r.horizon} steps</span></div>`;
    const bars = document.createElement('div'); bars.className = 'retbars';
    [['intuitor', 'INTUITOR', r.intuitor], ['ours', 'Ours', r.ours], ['grpo', 'GRPO-GT', r.grpo]].forEach(([k, lab, v]) => {
      const mean = (v[0] + v[1]) / 2;
      const b = document.createElement('div'); b.className = 'retbar';
      b.innerHTML = `<span class="rl">${lab}</span>
        <span class="rt"><i class="fill ${k}" style="--w:${Math.max(mean, 0.6)}%"></i></span>
        <span class="rv ${k}">${v[0].toFixed(1)} / ${v[1].toFixed(1)}</span>`;
      bars.appendChild(b);
    });
    row.appendChild(bars); host.appendChild(row);
  });
}

/* ---------------- 7B reveal ---------------- */
function buildSnapshot() {
  const tb = document.querySelector('#snap7b tbody');
  DATA.snapshot7b.forEach(r => {
    const tr = document.createElement('tr');
    const cell = (a, cls) => `<td class="num ${cls}">${a[0].toFixed(1)}<i>${a[1].toFixed(1)}</i><b>${a[2].toFixed(1)}</b></td>`;
    tr.innerHTML = `<th scope="row">${r.bench}<span class="kk">pass@${r.k}</span></th>` +
      cell(r.grpo, 'grpo') + cell(r.intuitor, 'intuitor zero') + cell(r.ours, 'ours');
    tb.appendChild(tr);
  });
}

/* ---------------- hero canvas ---------------- */
function hero() {
  const cv = document.getElementById('heroCanvas'); if (!cv) return;
  const ctx = cv.getContext('2d');
  const series = [
    { v: DATA.qwen3.gsm8k.grpo, c: 'rgba(170,182,198,.55)', w: 1.6, dash: [5, 4] },
    { v: DATA.qwen3.gsm8k.intuitor, c: 'rgba(224,122,88,.95)', w: 2.2 },
    { v: DATA.qwen3.gsm8k.ours, c: 'rgba(60,206,186,.95)', w: 2.6 }
  ];
  let w, h, dpr;
  const size = () => {
    dpr = Math.min(window.devicePixelRatio || 1, 2);
    w = cv.clientWidth; h = cv.clientHeight;
    cv.width = w * dpr; cv.height = h * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  };
  size(); window.addEventListener('resize', size);
  const n = series[0].v.length;
  const X = i => 40 + i / (n - 1) * (w - 80);
  const Y = v => h - 34 - (v - 48) / (90 - 48) * (h - 70);
  let t0 = performance.now();
  const frame = now => {
    const cycle = 5200, k = REDUCED ? 1 : Math.min(1, ((now - t0) % cycle) / 3000);
    ctx.clearRect(0, 0, w, h);
    ctx.strokeStyle = 'rgba(255,255,255,.055)'; ctx.lineWidth = 1;
    for (let g = 0; g <= 4; g++) { const y = 20 + g * (h - 54) / 4; ctx.beginPath(); ctx.moveTo(30, y); ctx.lineTo(w - 30, y); ctx.stroke(); }
    series.forEach(s => {
      ctx.save(); ctx.strokeStyle = s.c; ctx.lineWidth = s.w; ctx.lineJoin = 'round'; ctx.lineCap = 'round';
      if (s.dash) ctx.setLineDash(s.dash);
      const upto = k * (n - 1);
      ctx.beginPath();
      for (let i = 0; i <= Math.floor(upto); i++) { const x = X(i), y = Y(s.v[i]); i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }
      const f = upto - Math.floor(upto), i0 = Math.floor(upto);
      if (f > 0 && i0 < n - 1) {
        const x = X(i0) + (X(i0 + 1) - X(i0)) * f, y = Y(s.v[i0]) + (Y(s.v[i0 + 1]) - Y(s.v[i0])) * f;
        ctx.lineTo(x, y); ctx.stroke(); ctx.setLineDash([]);
        ctx.beginPath(); ctx.arc(x, y, 3.4, 0, 6.3); ctx.fillStyle = s.c; ctx.fill();
      } else ctx.stroke();
      ctx.restore();
    });
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

/* ---------------- qualitative loop ---------------- */
function qual() {
  const box = document.getElementById('loopbox'); if (!box) return;
  const lines = DATA.qual.intuitorLoop;
  let i = 0, reps = 0;
  const push = () => {
    const p = document.createElement('p');
    p.textContent = lines[i % lines.length];
    if (i % lines.length >= 2) p.className = 'rep';
    box.appendChild(p);
    box.scrollTop = box.scrollHeight;
    i++;
    if (i > 10) { box.innerHTML = ''; i = 0; reps++; }
  };
  if (REDUCED) { lines.forEach(l => { const p = document.createElement('p'); p.textContent = l; box.appendChild(p); }); return; }
  push(); setInterval(push, 1100);
}

/* ---------------- reveal on scroll ---------------- */
function reveals(replays) {
  const io = new IntersectionObserver(es => es.forEach(e => {
    if (!e.isIntersecting) return;
    e.target.classList.add('in');
    const id = e.target.dataset.replay;
    if (id && replays[id] && !e.target.dataset.done) { e.target.dataset.done = 1; replays[id](); }
    if (e.target.classList.contains('retwrap')) e.target.querySelectorAll('.fill').forEach(f => f.classList.add('go'));
    io.unobserve(e.target);
  }), { threshold: 0.25 });
  document.querySelectorAll('.reveal, .retwrap').forEach(n => io.observe(n));
}

document.addEventListener('DOMContentLoaded', () => {
  bgfx(); hero();
  const replays = { traj: buildTrajectory(), klcov: buildKlcov() };
  buildRetention(); buildSnapshot(); qual(); reveals(replays);
  const y = document.getElementById('yr'); if (y) y.textContent = new Date().getFullYear();
  document.querySelectorAll('[data-copy]').forEach(b => b.addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(document.getElementById(b.dataset.copy).textContent.trim()); const o = b.textContent; b.textContent = 'Copied'; setTimeout(() => b.textContent = o, 1400); } catch (e) { }
  }));
});

/* ---------------- animated background: entropy-collapse constellation ----------------
   A policy's token distribution, visualised. Nodes explore freely while entropy is high;
   as the run over-optimises its proxy they are dragged into one deterministic attractor,
   links shorten into a dense knot, colour shifts from CIRCA blue to failure coral — then
   the run resets and exploration returns. Soft orbs sit underneath so colour survives the
   panel blur even where the network does not. */
function bgfx() {
  const cv = document.getElementById('bgfx'); if (!cv) return;
  const ctx = cv.getContext('2d');
  let W = 0, H = 0, N = [], orbs = [], T = {};

  const isDark = () => {
    const a = document.documentElement.getAttribute('data-theme');
    return a ? a === 'dark' : window.matchMedia('(prefers-color-scheme: dark)').matches;
  };
  const rgb = v => {
    const s = getComputedStyle(document.documentElement).getPropertyValue(v).trim();
    const h = s.match(/^#?([0-9a-f]{6})$/i);
    if (h) { const n = parseInt(h[1], 16); return [n >> 16 & 255, n >> 8 & 255, n & 255]; }
    const r = s.match(/rgba?\(([^)]+)\)/); return r ? r[1].split(',').slice(0, 3).map(Number) : [120, 150, 200];
  };
  const mix = (a, b, k) => [0, 1, 2].map(i => Math.round(a[i] + (b[i] - a[i]) * k));

  const readTheme = () => {
    const d = isDark();
    T = { dark: d, ours: rgb('--ours'), fail: rgb('--fail'), amber: rgb('--amber'),
          node: d ? .95 : .72, link: d ? .40 : .26, orb: d ? .22 : .16 };
  };

  const spawn = () => {
    N = Array.from({ length: 58 }, () => ({
      x: Math.random() * W, y: Math.random() * H,
      vx: (Math.random() - .5) * .55, vy: (Math.random() - .5) * .55,
      r: 1.5 + Math.random() * 1.9, j: Math.random() * 6.283
    }));
    orbs = [
      { c: 'ours',  r: .50, x: .16, y: .20, sx: .052, sy: .037, p: 0 },
      { c: 'amber', r: .40, x: .85, y: .74, sx: .041, sy: .049, p: 2.1 },
      { c: 'fail',  r: .36, x: .72, y: .22, sx: .046, sy: .032, p: 4.3 }
    ];
  };

  const size = () => {
    const dp = Math.min(window.devicePixelRatio || 1, 2);
    W = window.innerWidth; H = window.innerHeight;
    cv.width = W * dp; cv.height = H * dp;
    cv.style.width = W + 'px'; cv.style.height = H + 'px';
    ctx.setTransform(dp, 0, 0, dp, 0, 0);
    spawn();
  };

  const smooth = k => k <= 0 ? 0 : k >= 1 ? 1 : k * k * (3 - 2 * k);
  const PERIOD = 30;                                   // one full rise-and-fall, seconds

  // entropy over the cycle: explore → collapse → held degenerate → respawn
  const entropyAt = t => {
    const c = (t % PERIOD) / PERIOD;
    if (c < .38) return 1;                              // exploring
    if (c < .70) return 1 - smooth((c - .38) / .32);    // collapsing
    if (c < .84) return 0;                              // collapsed, deterministic
    return smooth((c - .84) / .16);                     // reset, exploration returns
  };

  const draw = t => {
    const ent = entropyAt(t), col = mix(T.ours, T.fail, 1 - ent);
    // attractor: the degenerate fixed point the policy falls into
    const ax = W * (.5 + Math.sin(t * .045) * .16), ay = H * (.5 + Math.cos(t * .037) * .14);

    ctx.clearRect(0, 0, W, H);

    // --- soft colour field (survives the panel blur) ---
    ctx.globalCompositeOperation = T.dark ? 'lighter' : 'source-over';
    orbs.forEach(o => {
      const base = o.c === 'ours' ? T.ours : o.c === 'fail' ? T.fail : T.amber;
      const c3 = o.c === 'ours' ? mix(base, T.fail, (1 - ent) * .85) : base;
      const x = (o.x + Math.sin(t * o.sx + o.p) * .09) * W;
      const y = (o.y + Math.cos(t * o.sy + o.p * 1.7) * .08) * H;
      const r = o.r * Math.min(W, H) * (1 + Math.sin(t * .03 + o.p) * .07) * (.75 + ent * .25);
      const g = ctx.createRadialGradient(x, y, 0, x, y, r);
      g.addColorStop(0, `rgba(${c3},${T.orb})`);
      g.addColorStop(.55, `rgba(${c3},${T.orb * .34})`);
      g.addColorStop(1, `rgba(${c3},0)`);
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, r, 0, 6.2832); ctx.fill();
    });
    ctx.globalCompositeOperation = 'source-over';

    // --- physics: free drift at high entropy, pulled to one point as it falls ---
    const pull = (1 - ent) * (1 - ent) * .0022, damp = .996 - (1 - ent) * .045, sp = .32 + ent * .68;
    for (const n of N) {
      n.vx += (ax - n.x) * pull; n.vy += (ay - n.y) * pull;
      if (ent > .02) {                                  // exploration jitter dies with entropy
        n.vx += (Math.random() - .5) * .05 * ent;
        n.vy += (Math.random() - .5) * .05 * ent;
      }
      n.vx *= damp; n.vy *= damp;
      n.x += n.vx * sp; n.y += n.vy * sp;
      if (n.x < -20) n.x = W + 20; if (n.x > W + 20) n.x = -20;
      if (n.y < -20) n.y = H + 20; if (n.y > H + 20) n.y = -20;
    }

    // --- links ---
    const D = 118 + ent * 34, la = T.link;
    ctx.lineWidth = T.dark ? 1.15 : 1;
    for (let i = 0; i < N.length; i++) {
      const a = N[i];
      for (let j = i + 1; j < N.length; j++) {
        const b = N[j], dx = a.x - b.x, dy = a.y - b.y, d2 = dx * dx + dy * dy;
        if (d2 > D * D) continue;
        const k = 1 - Math.sqrt(d2) / D;
        ctx.strokeStyle = `rgba(${col},${k * la})`;
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      }
    }

    // --- nodes ---
    for (const n of N) {
      const pulse = 1 + Math.sin(t * 1.6 + n.j) * .14 * ent;
      ctx.fillStyle = `rgba(${col},${T.node})`;
      ctx.beginPath(); ctx.arc(n.x, n.y, n.r * pulse * (.7 + ent * .3), 0, 6.2832); ctx.fill();
    }

    // --- collapse halo: the degenerate fixed point, visible only while collapsed ---
    if (ent < .5) {
      const k = 1 - ent * 2, rr = 26 + (1 - ent) * 60 + Math.sin(t * 2.1) * 6;
      const g = ctx.createRadialGradient(ax, ay, 0, ax, ay, rr);
      g.addColorStop(0, `rgba(${T.fail},${.20 * k})`);
      g.addColorStop(1, `rgba(${T.fail},0)`);
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(ax, ay, rr, 0, 6.2832); ctx.fill();
    }
  };

  const frame = now => { draw(now / 1000); if (!document.hidden) requestAnimationFrame(frame); };

  readTheme(); size();
  window.addEventListener('resize', size);
  new MutationObserver(readTheme).observe(document.documentElement,
    { attributes: true, attributeFilter: ['data-theme'] });
  window.matchMedia('(prefers-color-scheme: dark)').addEventListener('change', readTheme);

  if (REDUCED) {                                        // one static frame mid-exploration
    draw(PERIOD * .2);
    window.addEventListener('resize', () => { size(); draw(PERIOD * .2); });
    new MutationObserver(() => { readTheme(); draw(PERIOD * .2); })
      .observe(document.documentElement, { attributes: true, attributeFilter: ['data-theme'] });
    return;
  }
  requestAnimationFrame(frame);
  document.addEventListener('visibilitychange', () => { if (!document.hidden) requestAnimationFrame(frame); });
}
