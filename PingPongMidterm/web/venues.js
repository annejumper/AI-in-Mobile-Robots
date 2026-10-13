// T-anne-is - the three stadiums.
//
// Each surface gets a venue *in the style of* a Grand Slam (no real logos or
// sponsor names): Paris clay in afternoon sun, London grass in soft daylight,
// and a New York hard court night session. Everything is built from code --
// no image files -- so it stays sharp from any angle.

import * as THREE from 'three';

// --- court geometry (same numbers as court.py), python world -> three.js ------------
export const BASELINE_Z = 1.0, NET_Z = BASELINE_Z + 11.885, FAR_Z = NET_Z + 11.885, OPP_Z = FAR_Z + 1.0;
export const HALF_S = 4.115, HALF_D = 5.485, SERVICE = 6.40, NET_H = 0.914, POST_H = 1.07, POST_X = HALF_D + 0.914;
export const SIDE_WALL_X = HALF_D + 3.7, FAR_WALL_Z = FAR_Z + 6.4;
export const V = (x, y, z) => new THREE.Vector3(x, y, -z);

// --- texture helpers ------------------------------------------------------------------
let ANISO = 8;
const QUALITY = { crowd: 0.9 };

export function canvasTex(w, h, draw, repeat) {
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  draw(c.getContext('2d'), w, h);
  const t = new THREE.CanvasTexture(c);
  t.colorSpace = THREE.SRGBColorSpace;
  t.anisotropy = ANISO;
  if (repeat) { t.wrapS = t.wrapT = THREE.RepeatWrapping; t.repeat.set(repeat[0], repeat[1]); }
  return t;
}

export function addNoise(ctx, w, h, amount) {
  const img = ctx.getImageData(0, 0, w, h), d = img.data;
  for (let i = 0; i < d.length; i += 4) {
    const n = (Math.random() - 0.5) * amount * 2;
    d[i] += n; d[i + 1] += n; d[i + 2] += n * 0.8;
  }
  ctx.putImageData(img, 0, 0);
}

function blotches(c, w, h, n, rMin, rMax, colors) {
  // soft round patches, drawn wrapped so a repeating texture has no seams
  for (let i = 0; i < n; i++) {
    const x = Math.random() * w, y = Math.random() * h, r = rMin + Math.random() * (rMax - rMin);
    const col = colors[(Math.random() * colors.length) | 0];
    for (const ox of [-w, 0, w]) for (const oy of [-h, 0, h]) {
      const g = c.createRadialGradient(x + ox, y + oy, 0, x + ox, y + oy, r);
      g.addColorStop(0, col); g.addColorStop(1, 'rgba(0,0,0,0)');
      c.fillStyle = g; c.fillRect(x + ox - r, y + oy - r, 2 * r, 2 * r);
    }
  }
}

const rnd = (a, b) => a + Math.random() * (b - a);
const gauss = () => (Math.random() + Math.random() + Math.random() - 1.5) / 1.5;

// --- the three venues ---------------------------------------------------------------------
export const VENUES = {
  clay: {
    title: 'Roland Garros', mark: { text: 'ROLAND GARROS', color: '#ffffff', accent: '#e8692c', hi: '#f08a4b' },
    sky: [0x2f7fd6, 0xeaf0f4], fog: [0xe8e1d6, 90, 230],
    hemi: [0xe2efff, 0xa4674a, 1.0], sun: { color: 0xffdcb0, intensity: 3.1, pos: [-11, 30, 15] },
    exposure: 1.0, bloom: 0.22,
    wall: '#0b4a36', wallText: '#ffffff', trim: '#ffffff', seat: '#1d6b45', post: 0x0b4a36,
    shirt: 0x1fa6a0, accent: '#e07a45',
    crowd: ['#f2f2f2', '#1e3a5f', '#d94f3a', '#e8c34e', '#7aa6c2', '#2e2e2e', '#c98f78', '#4f7942', '#8a5a9e', '#3b3b4f'],
    fx: { dust: [0.80, 0.45, 0.27], dustN: 24, mark: '#7a3618', bounce: { freq: 110, noise: 0.9, nfreq: 650 } },
  },
  grass: {
    title: 'Wimbledon', mark: { text: 'WIMBLEDON', color: '#efe6c8', accent: '#6b3fa0', hi: '#c9a8ff' },
    sky: [0x7aa3cf, 0xe6edf2], fog: [0xe1e7ea, 90, 230],
    hemi: [0xeaf2ff, 0x4f6a3e, 1.3], sun: { color: 0xfff7ea, intensity: 2.2, pos: [14, 34, 8] },
    exposure: 1.0, bloom: 0.18,
    wall: '#1c4a30', wallText: '#efe6c8', trim: '#4b2a6b', seat: '#1b4a2c', post: 0x1c4a30,
    shirt: 0xffffff, accent: '#7b4fb0',
    crowd: ['#f4f1e6', '#2d3e50', '#7d9cc0', '#c98aa6', '#3f5e3a', '#e0c96a', '#9a4a4a', '#ffffff', '#5b4a7a', '#b8b0a0', '#1f2a3a'],
    fx: { dust: [0.42, 0.72, 0.33], dustN: 8, mark: null, bounce: { freq: 90, noise: 0.5, nfreq: 420 } },
  },
  hard: {
    title: 'US Open', mark: { text: 'US OPEN', color: '#ffffff', accent: '#ffd23f', hi: '#7fb6ff' },
    sky: [0x03060f, 0x17213f], fog: [0x0a1122, 110, 260],
    hemi: [0x9fb4e0, 0x1b2a22, 0.6], sun: { color: 0xf0f4ff, intensity: 2.6, pos: [2, 46, 4] },
    exposure: 1.12, bloom: 0.55,
    wall: '#13213d', wallText: '#ffffff', trim: '#7fb6ff', seat: '#1e3c78', post: 0x13213d,
    shirt: 0xff3b6b, accent: '#4a8fe0',
    crowd: ['#1a1a1a', '#2c3e50', '#ffffff', '#c0392b', '#f1c40f', '#3d5a80', '#8e44ad', '#16a085', '#555555'],
    fx: { dust: null, dustN: 0, mark: null, bounce: { freq: 160, noise: 0.8, nfreq: 1600 } },
  },
};

// --- surfaces -----------------------------------------------------------------------------
function clayTex() {
  return canvasTex(1024, 1024, (c, w, h) => {
    c.fillStyle = '#bf6038'; c.fillRect(0, 0, w, h);
    blotches(c, w, h, 220, 30, 160, ['rgba(232,150,100,0.10)', 'rgba(120,45,20,0.09)', 'rgba(210,120,80,0.08)']);
    // brushed sweep arcs left by the court drag mats
    c.lineCap = 'round';
    for (let i = 0; i < 70; i++) {
      c.strokeStyle = Math.random() < 0.5 ? 'rgba(255,205,170,0.05)' : 'rgba(110,40,15,0.05)';
      c.lineWidth = rnd(4, 14);
      c.beginPath(); c.arc(rnd(0, w), rnd(0, h), rnd(300, 900), rnd(0, 6.3), rnd(0, 6.3) + 0.6); c.stroke();
    }
    addNoise(c, w, h, 22);
  }, [7, 10]);
}

function clayScuffTex() {
  // footwork slides and scuffs around the baseline
  return canvasTex(512, 256, (c, w, h) => {
    c.clearRect(0, 0, w, h);
    c.lineCap = 'round';
    for (let i = 0; i < 260; i++) {
      const x = w / 2 + gauss() * w * 0.45, y = h * 0.55 + gauss() * h * 0.5;
      c.strokeStyle = Math.random() < 0.6 ? `rgba(120,42,16,${rnd(0.08, 0.22)})` : `rgba(240,170,125,${rnd(0.08, 0.2)})`;
      c.lineWidth = rnd(3, 9);
      const a = rnd(-0.6, 0.6) + (Math.random() < 0.5 ? 0 : Math.PI / 2), l = rnd(8, 40);
      c.beginPath(); c.moveTo(x, y); c.lineTo(x + Math.cos(a) * l, y + Math.sin(a) * l); c.stroke();
    }
  });
}

function grassCourtTex(apron) {
  return canvasTex(512, 2048, (c, w, h) => {
    const bands = apron ? 10 : 18;
    for (let i = 0; i < bands; i++) {
      c.fillStyle = i % 2 ? (apron ? '#3f8338' : '#4f9444') : (apron ? '#478d3f' : '#5ea851');
      c.fillRect(0, (i * h) / bands, w, h / bands + 1);
    }
    blotches(c, w, h, 160, 10, 60, ['rgba(30,70,20,0.10)', 'rgba(150,200,110,0.08)']);
    if (!apron) {
      // week-two wear: brown, patchy, concentrated just behind each baseline
      const margin = 3 / (FAR_Z - BASELINE_Z + 6);  // where the baselines sit in the texture
      for (const by of [margin, 1 - margin]) {
        for (let i = 0; i < 500; i++) {
          const x = w / 2 + gauss() * w * 0.11, y = (by + Math.abs(gauss()) * 0.03 * (by < 0.5 ? -1 : 1) + gauss() * 0.006) * h;
          c.fillStyle = Math.random() < 0.7 ? `rgba(150,120,72,${rnd(0.12, 0.32)})` : `rgba(185,170,100,${rnd(0.08, 0.22)})`;
          c.beginPath(); c.arc(x, y, rnd(3, 9), 0, Math.PI * 2); c.fill();
        }
      }
    }
    addNoise(c, w, h, 12);
  });
}

function hardTex(hex, repeat) {
  return canvasTex(512, 512, (c, w, h) => {
    c.fillStyle = hex; c.fillRect(0, 0, w, h);
    blotches(c, w, h, 60, 40, 140, ['rgba(255,255,255,0.025)', 'rgba(0,0,0,0.03)']);
    addNoise(c, w, h, 7);
  }, repeat);
}

// --- ad boards: the tournament, the game, and its one sponsor --------------------------------
const FONT = '-apple-system, "Helvetica Neue", Helvetica, Arial, sans-serif';

function drawBall(c, x, y, r) {
  c.fillStyle = '#d8f23e'; c.beginPath(); c.arc(x, y, r, 0, Math.PI * 2); c.fill();
  c.strokeStyle = '#ffffff'; c.lineWidth = r * 0.16;
  c.beginPath(); c.arc(x - r * 1.25, y, r * 0.95, -0.75, 0.75); c.stroke();
  c.beginPath(); c.arc(x + r * 1.25, y, r * 0.95, Math.PI - 0.75, Math.PI + 0.75); c.stroke();
}

function drawAd(c, item, v, cx, cy, h) {
  // one ad panel centered at (cx, cy)
  c.textAlign = 'center'; c.textBaseline = 'middle';
  if (item === 'venue') {
    c.font = `800 ${h * 0.42}px ${FONT}`; c.fillStyle = v.mark.color;
    c.fillText(v.mark.text, cx, cy);
    const tw = c.measureText(v.mark.text).width;
    c.fillStyle = v.mark.accent; c.fillRect(cx - tw / 2, cy + h * 0.27, tw, h * 0.06);
  } else if (item === 'sponsor') {
    c.font = `italic 800 ${h * 0.42}px ${FONT}`;
    const t = 'Tennis > Ping Pong', tw = c.measureText(t).width;
    drawBall(c, cx - tw / 2 - h * 0.18, cy, h * 0.15);
    c.fillStyle = '#ffd23f'; c.fillText(t, cx + h * 0.12, cy);
  } else if (item === 'game') {
    c.font = `800 ${h * 0.44}px ${FONT}`;
    const parts = [['T-', v.wallText], ['anne', v.mark.hi], ['-is', v.wallText]];
    const total = parts.reduce((s, [t]) => s + c.measureText(t).width, 0);
    let x = cx - total / 2;
    c.textAlign = 'left';
    for (const [t, col] of parts) {
      c.fillStyle = col; c.fillText(t, x, cy);
      x += c.measureText(t).width;
    }
  } else {
    c.font = `600 ${h * 0.3}px ${FONT}`; c.fillStyle = v.wallText;
    c.fillText(item, cx, cy);
  }
}

function adTex(v, items, panels, repeat) {
  // a strip of ad panels on the venue's wall color, with a trim line on top
  const pw = 680, h = 160;
  return canvasTex(pw * panels, h, (c, w) => {
    c.fillStyle = v.wall; c.fillRect(0, 0, w, h);
    c.fillStyle = v.trim; c.fillRect(0, 0, w, 9);
    for (let i = 0; i < panels; i++) {
      if (i) { c.fillStyle = 'rgba(255,255,255,0.18)'; c.fillRect(i * pw - 1, 24, 2, h - 40); }
      drawAd(c, items[i % items.length], v, (i + 0.5) * pw, h / 2 + 6, h);
    }
  }, repeat);
}

function ledTex(v) {
  // the glowing ribbon board between the tiers of the night stadium
  return canvasTex(680 * 4, 128, (c, w, h) => {
    c.fillStyle = '#04060c'; c.fillRect(0, 0, w, h);
    ['sponsor', 'venue', 'game', 'sponsor'].forEach((it, i) =>
      drawAd(c, it, { ...v, wallText: '#ffffff' }, (i + 0.5) * 680, h / 2 + 4, h));
    c.fillStyle = 'rgba(255,255,255,0.06)';                   // LED pixel grid
    for (let x = 0; x < w; x += 4) c.fillRect(x, 0, 1, h);
  }, [3, 1]);
}

function shadeTex() {
  // a soft dark gradient used as a fake contact shadow along walls
  return canvasTex(8, 128, (c, w, h) => {
    const g = c.createLinearGradient(0, 0, 0, h);
    g.addColorStop(0, 'rgba(0,0,0,0.42)'); g.addColorStop(1, 'rgba(0,0,0,0)');
    c.fillStyle = g; c.fillRect(0, 0, w, h);
  });
}

function plane(w, l, mat, x, y, z, rotZ = 0) {
  const m = new THREE.Mesh(new THREE.PlaneGeometry(w, l), mat);
  m.rotation.x = -Math.PI / 2;
  m.rotation.z = rotZ;
  m.position.copy(V(x, y, z));
  m.receiveShadow = true;
  return m;
}

function buildSurface(g, key) {
  const mid = (BASELINE_Z + FAR_Z) / 2, len = FAR_Z - BASELINE_Z;
  if (key === 'clay') {
    g.add(plane(70, 110, new THREE.MeshStandardMaterial({ map: clayTex(), roughness: 1 }), 0, 0, mid));
    const scuff = new THREE.MeshStandardMaterial({ map: clayScuffTex(), transparent: true, depthWrite: false, roughness: 1, polygonOffset: true, polygonOffsetFactor: -1 });
    g.add(plane(7, 3.5, scuff, 0, 0.002, BASELINE_Z - 0.5));
    g.add(plane(7, 3.5, scuff, 0, 0.002, FAR_Z + 0.5, Math.PI));
  } else if (key === 'grass') {
    g.add(plane(70, 110, new THREE.MeshStandardMaterial({ map: grassCourtTex(true), roughness: 0.9 }), 0, 0, mid));
    g.add(plane(2 * HALF_D + 3, len + 6, new THREE.MeshStandardMaterial({ map: grassCourtTex(false), roughness: 0.85 }), 0, 0.002, mid));
  } else {
    g.add(plane(70, 110, new THREE.MeshStandardMaterial({ map: hardTex('#5f8f5a', [20, 30]), roughness: 0.55 }), 0, 0, mid));
    g.add(plane(2 * HALF_D + 3.6, len + 7.5, new THREE.MeshStandardMaterial({ map: hardTex('#3a679f', [5, 10]), roughness: 0.42, envMapIntensity: 1.3 }), 0, 0.002, mid));
  }

  const lineMat = new THREE.MeshStandardMaterial({
    color: key === 'clay' ? 0xf1ebe3 : 0xf7f7f2, roughness: key === 'hard' ? 0.35 : 0.8,
    polygonOffset: true, polygonOffsetFactor: -2,
  });
  const rect = (x1, z1, x2, z2) => g.add(plane(Math.abs(x2 - x1), Math.abs(z2 - z1), lineMat, (x1 + x2) / 2, 0.004, (z1 + z2) / 2));
  const L = 0.05, B = 0.10;
  rect(-HALF_D, BASELINE_Z - B, HALF_D, BASELINE_Z);
  rect(-HALF_D, FAR_Z, HALF_D, FAR_Z + B);
  for (const sx of [-1, 1]) {
    rect(sx * HALF_D - L / 2, BASELINE_Z - B, sx * HALF_D + L / 2, FAR_Z + B);
    rect(sx * HALF_S - L / 2, BASELINE_Z - B, sx * HALF_S + L / 2, FAR_Z + B);
  }
  for (const zs of [NET_Z - SERVICE, NET_Z + SERVICE]) rect(-HALF_S, zs - L / 2, HALF_S, zs + L / 2);
  rect(-L / 2, NET_Z - SERVICE, L / 2, NET_Z + SERVICE);
  rect(-L / 2, BASELINE_Z, L / 2, BASELINE_Z + 0.15);
  rect(-L / 2, FAR_Z - 0.15, L / 2, FAR_Z);
}

// --- walls, stands, roof ------------------------------------------------------------------
function buildWalls(g, v) {
  const wallMat = new THREE.MeshStandardMaterial({ map: adTex(v, ['venue', 'sponsor', 'game'], 3), roughness: 0.75 });
  const far = new THREE.Mesh(new THREE.BoxGeometry(2 * SIDE_WALL_X + 0.4, 1.4, 0.3), wallMat);
  far.position.copy(V(0, 0.7, FAR_WALL_Z)); g.add(far);
  const sideMat = new THREE.MeshStandardMaterial({ color: v.wall, roughness: 0.75 });
  for (const sx of [-1, 1]) {
    const w = new THREE.Mesh(new THREE.BoxGeometry(0.3, 1.1, FAR_WALL_Z + 10), sideMat);
    w.position.copy(V(sx * SIDE_WALL_X, 0.55, (FAR_WALL_Z - 10) / 2)); w.castShadow = true;
    g.add(w);
    // ad boards along the inside of each side wall
    const len = FAR_WALL_Z + 10, ads = new THREE.MeshStandardMaterial({
      map: adTex(v, ['sponsor', 'venue', 'game', 'ME193 \u00b7 ANNE JUMPER'], 4, [len / 24, 1]), roughness: 0.7 });
    const board = new THREE.Mesh(new THREE.PlaneGeometry(len, 1.1), ads);
    board.rotation.y = sx < 0 ? Math.PI / 2 : -Math.PI / 2;     // facing the court, reading left to right
    board.position.copy(V(sx * (SIDE_WALL_X - 0.16), 0.55, (FAR_WALL_Z - 10) / 2));
    g.add(board);
  }
  // fake contact shadows where the walls meet the court
  const shade = new THREE.MeshBasicMaterial({ map: shadeTex(), transparent: true, depthWrite: false });
  g.add(plane(2 * SIDE_WALL_X, 1.2, shade, 0, 0.005, FAR_WALL_Z - 0.75, Math.PI));
  for (const sx of [-1, 1]) {
    g.add(plane(FAR_WALL_Z + 10, 1.0, shade, sx * (SIDE_WALL_X - 0.65), 0.005, (FAR_WALL_Z - 10) / 2, sx * Math.PI / 2));
  }
}

function buildBowl(g, v, tiers, roof) {
  // A rectangular bowl on three sides (far end + both sides), row by row,
  // with an instanced spectator in most seats.
  const innerX = SIDE_WALL_X + 1.4, innerZ = FAR_WALL_Z + 1.4, nearZ = -14;
  const rows = [], bands = [];
  let y = 1.5, d = 0;
  for (const t of tiers) {
    if (t.band) { bands.push({ y, d, h: t.band, depth: 1.6 }); y += t.band; d += 1.6; continue; }
    for (let r = 0; r < t.rows; r++) { rows.push({ y, d, depth: t.depth }); y += t.rise; d += t.depth; }
  }

  const unit = new THREE.BoxGeometry(1, 1, 1);
  const seatMat = new THREE.MeshStandardMaterial({ color: v.seat, roughness: 0.8 });
  const blocks = new THREE.InstancedMesh(unit, seatMat, rows.length * 3);
  const m = new THREE.Matrix4(), q = new THREE.Quaternion(), s = new THREE.Vector3(), p = new THREE.Vector3();
  let bi = 0;
  const put = (mesh, i, x, yy, z, sx, sy, sz) => { mesh.setMatrixAt(i, m.compose(p.copy(V(x, yy, z)), q, s.set(sx, sy, sz))); };
  for (const r of rows) {
    const halfW = innerX + r.d + r.depth;
    put(blocks, bi++, 0, r.y / 2, innerZ + r.d + r.depth / 2, 2 * halfW, r.y, r.depth);
    const zLen = innerZ + r.d + r.depth - nearZ;
    for (const sx of [-1, 1]) put(blocks, bi++, sx * (innerX + r.d + r.depth / 2), r.y / 2, nearZ + zLen / 2, r.depth, r.y, zLen);
  }
  blocks.receiveShadow = true;
  g.add(blocks);

  // spectators
  const seats = [];
  for (const r of rows) {
    const halfW = innerX + r.d;
    for (let x = -halfW + 0.3; x < halfW - 0.3; x += 0.56) seats.push([x + rnd(-0.05, 0.05), r.y, innerZ + r.d + r.depth * 0.45]);
    for (const sx of [-1, 1]) {
      for (let z = nearZ + 1; z < innerZ + r.d - 0.3; z += 0.56) seats.push([sx * (innerX + r.d + r.depth * 0.45), r.y, z + rnd(-0.05, 0.05)]);
    }
  }
  const filled = seats.filter(() => Math.random() < QUALITY.crowd);
  const bodyGeo = new THREE.BoxGeometry(0.42, 0.55, 0.3), headGeo = new THREE.SphereGeometry(0.11, 8, 6);
  const bodies = new THREE.InstancedMesh(bodyGeo, new THREE.MeshStandardMaterial({ roughness: 0.9 }), filled.length);
  const heads = new THREE.InstancedMesh(headGeo, new THREE.MeshStandardMaterial({ roughness: 0.8 }), filled.length);
  const col = new THREE.Color();
  filled.forEach(([x, yy, z], i) => {
    put(bodies, i, x, yy + 0.3, z, 1, 1, 1);
    put(heads, i, x, yy + 0.72 + rnd(-0.03, 0.05), z, 1, 1, 1);
    bodies.setColorAt(i, col.set(v.crowd[(Math.random() * v.crowd.length) | 0]).offsetHSL(0, 0, rnd(-0.06, 0.06)));
    // skin tones, plus hair and hats seen from the court
    const r = Math.random();
    heads.setColorAt(i, r < 0.55 ? col.setHSL(rnd(0.04, 0.08), rnd(0.3, 0.5), rnd(0.2, 0.55))
      : r < 0.85 ? col.setHSL(rnd(0.05, 0.1), rnd(0.2, 0.5), rnd(0.06, 0.3))
      : col.set(v.crowd[(Math.random() * v.crowd.length) | 0]));
  });
  g.add(bodies, heads);

  // the band between tiers (night stadium suites)
  for (const b of bands) {
    const led = ledTex(v);
    const mat = new THREE.MeshStandardMaterial({ map: led, emissive: 0xffffff, emissiveMap: led, emissiveIntensity: 1.1, roughness: 0.3 });
    const halfW = innerX + b.d + b.depth;
    const f = new THREE.Mesh(new THREE.BoxGeometry(2 * halfW, b.h, b.depth), mat);
    f.position.copy(V(0, b.y + b.h / 2, innerZ + b.d + b.depth / 2)); g.add(f);
    const zLen = innerZ + b.d + b.depth - nearZ;
    for (const sx of [-1, 1]) {
      const sm = new THREE.Mesh(new THREE.BoxGeometry(b.depth, b.h, zLen), mat);
      sm.position.copy(V(sx * (innerX + b.d + b.depth / 2), b.y + b.h / 2, nearZ + zLen / 2)); g.add(sm);
    }
  }

  // back wall + roof ring above the last row
  const top = y, outer = d;
  const backMat = new THREE.MeshStandardMaterial({ color: roof ? roof.color : 0x9aa0a6, roughness: 0.7 });
  const halfW = innerX + outer + 1;
  const back = new THREE.Mesh(new THREE.BoxGeometry(2 * halfW, 2.2, 0.6), backMat);
  back.position.copy(V(0, top + 1.1, innerZ + outer + 0.3)); g.add(back);
  for (const sx of [-1, 1]) {
    const zLen = innerZ + outer - nearZ;
    const b = new THREE.Mesh(new THREE.BoxGeometry(0.6, 2.2, zLen), backMat);
    b.position.copy(V(sx * (innerX + outer + 0.3), top + 1.1, nearZ + zLen / 2)); g.add(b);
  }
  if (roof) {
    // a canopy reaching in over the top rows, like a stadium with an open retractable roof
    const roofMat = new THREE.MeshStandardMaterial({ color: roof.color, roughness: 0.6, metalness: 0.2, side: THREE.DoubleSide, transparent: roof.opacity < 1, opacity: roof.opacity });
    const reach = roof.reach, ry = top + 3.2;
    const farR = new THREE.Mesh(new THREE.BoxGeometry(2 * halfW, 0.35, reach), roofMat);
    farR.position.copy(V(0, ry, innerZ + outer - reach / 2)); g.add(farR);
    for (const sx of [-1, 1]) {
      const zLen = innerZ + outer - nearZ;
      const sr = new THREE.Mesh(new THREE.BoxGeometry(reach, 0.35, zLen), roofMat);
      sr.position.copy(V(sx * (innerX + outer - reach / 2), ry, nearZ + zLen / 2)); g.add(sr);
    }
    // columns holding it up
    for (let x = -halfW + 2; x <= halfW - 2; x += 8) {
      const c = new THREE.Mesh(new THREE.CylinderGeometry(0.15, 0.15, 3.2, 8), backMat);
      c.position.copy(V(x, top + 1.6, innerZ + outer + 0.1)); g.add(c);
    }
    if (roof.lights) {
      // floodlights along the inner edge of the roof ring (they glow with bloom)
      const lightMat = new THREE.MeshBasicMaterial({ color: 0xffffff, toneMapped: false });
      lightMat.color.multiplyScalar(4);
      const lz = innerZ + outer - reach + 0.4;
      for (let x = -halfW + 3; x < halfW - 2; x += 2.6) {
        const l = new THREE.Mesh(new THREE.BoxGeometry(1.4, 0.18, 0.5), lightMat);
        l.position.copy(V(x, ry - 0.3, lz)); g.add(l);
      }
      for (const sx of [-1, 1]) {
        for (let z = nearZ + 4; z < innerZ + outer - reach; z += 2.6) {
          const l = new THREE.Mesh(new THREE.BoxGeometry(0.5, 0.18, 1.4), lightMat);
          l.position.copy(V(sx * (innerX + outer - reach + 0.4), ry - 0.3, z)); g.add(l);
        }
      }
    }
  }
  // how far out (from the court's center line) the side stands are at height h
  const levels = [...rows, ...bands].sort((a, b) => a.y - b.y);
  const sideXAt = h => innerX + (levels.find(r => r.y >= h) || { d: outer }).d;
  return { top, outer, innerZ, innerX, sideXAt };
}

// --- courtside ------------------------------------------------------------------------------
function buildCourtside(g, v) {
  const paint = new THREE.MeshStandardMaterial({ color: v.wall, roughness: 0.6 });
  const white = new THREE.MeshStandardMaterial({ color: 0xf2f2f2, roughness: 0.7 });
  const skin = new THREE.MeshStandardMaterial({ color: 0xd9a77f, roughness: 0.8 });
  const cx = POST_X + 1.4;

  // umpire's chair beside the net
  const chair = new THREE.Group();
  for (const [dx, dz] of [[-0.3, -0.3], [0.3, -0.3], [-0.3, 0.3], [0.3, 0.3]]) {
    const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 1.9, 6), paint);
    leg.position.set(dx, 0.95, dz); chair.add(leg);
  }
  const seat = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.12, 0.8), paint); seat.position.y = 1.95; chair.add(seat);
  const backrest = new THREE.Mesh(new THREE.BoxGeometry(0.8, 0.7, 0.08), paint); backrest.position.set(0, 2.35, 0); backrest.rotation.y = 0; chair.add(backrest);
  const ump = new THREE.Mesh(new THREE.CapsuleGeometry(0.2, 0.4, 4, 10), white); ump.position.set(0, 2.35, -0.15); chair.add(ump);
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.11, 12, 8), skin); head.position.set(0, 2.85, -0.15); chair.add(head);
  chair.position.copy(V(cx, 0, NET_Z));
  chair.rotation.y = -Math.PI / 2;               // facing across the court
  g.add(chair);

  // players' benches on the chair side
  for (const dz of [-2.6, 2.6]) {
    const bench = new THREE.Mesh(new THREE.BoxGeometry(0.6, 0.5, 1.6), paint);
    bench.position.copy(V(cx + 0.4, 0.25, NET_Z + dz)); g.add(bench);
    const towel = new THREE.Mesh(new THREE.BoxGeometry(0.62, 0.06, 0.5), white);
    towel.position.copy(V(cx + 0.4, 0.53, NET_Z + dz)); g.add(towel);
  }
  g.traverse(o => { if (o.isMesh && o.parent === chair) o.castShadow = true; });
}

// --- video scoreboard ------------------------------------------------------------------------
function buildBoard(g, v, key, bowl) {
  const c = document.createElement('canvas'); c.width = 1024; c.height = 320;
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace;
  const mat = new THREE.MeshBasicMaterial({ map: tex, toneMapped: false });
  if (key === 'hard') mat.color.multiplyScalar(1.6);   // LED board glows at night
  // A scoreboard tower just behind the far wall, in front of the crowd: high enough
  // to clear the opponent, low enough to sit under the roof and inside the camera view.
  const BOARD_Y = 9.6, boardZ = bowl.innerZ - 0.7;
  const board = new THREE.Mesh(new THREE.PlaneGeometry(9, 2.8), mat);
  board.position.copy(V(0, BOARD_Y, boardZ));
  const darkMat = new THREE.MeshStandardMaterial({ color: 0x15171c, roughness: 0.5, metalness: 0.3 });
  const frame = new THREE.Mesh(new THREE.BoxGeometry(9.5, 3.3, 0.4), darkMat);
  frame.position.copy(board.position).add(new THREE.Vector3(0, 0, -0.25));
  g.add(frame, board);

  // hung from a steel truss that spans the stadium between the two side stands
  const trussY = BOARD_Y + 3.6, span = 2 * bowl.sideXAt(trussY) + 1;
  const steel = new THREE.MeshStandardMaterial({ color: 0x8a929c, roughness: 0.4, metalness: 0.5 });
  for (const dy of [0.3, -0.3]) {                          // top and bottom chords
    const chord = new THREE.Mesh(new THREE.BoxGeometry(span, 0.12, 0.12), steel);
    chord.position.copy(V(0, trussY + dy, boardZ + 0.25)); g.add(chord);
  }
  if (key === 'hard') {                                     // night: an LED strip along the truss
    const led = new THREE.MeshBasicMaterial({ color: 0x9fd0ff, toneMapped: false });
    led.color.multiplyScalar(2.5);
    const strip = new THREE.Mesh(new THREE.BoxGeometry(span, 0.06, 0.06), led);
    strip.position.copy(V(0, trussY - 0.38, boardZ + 0.15)); g.add(strip);
  }
  for (let x = -span / 2, k = 0; x < span / 2; x += 0.6, k++) {   // zig-zag lacing
    const lace = new THREE.Mesh(new THREE.BoxGeometry(0.05, 0.72, 0.05), steel);
    lace.position.copy(V(x + 0.3, trussY, boardZ + 0.25));
    lace.rotation.z = k % 2 ? 0.45 : -0.45; g.add(lace);
  }
  const cableTop = trussY - 0.3, cableBottom = BOARD_Y + 1.65, cableLen = cableTop - cableBottom;
  for (const sx of [-4.2, -1.4, 1.4, 4.2]) {
    const cable = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.025, cableLen, 6), steel);
    cable.position.copy(V(sx, cableBottom + cableLen / 2, boardZ + 0.25)); g.add(cable);
  }

  const draw = (streak, record) => {
    const ctx = c.getContext('2d');
    ctx.fillStyle = key === 'grass' ? '#1c4a30' : key === 'clay' ? '#0b3d2c' : '#05070d';
    ctx.fillRect(0, 0, c.width, c.height);
    ctx.fillStyle = key === 'grass' ? '#efe6c8' : '#ffffff';
    ctx.font = 'bold 54px -apple-system, Helvetica, Arial'; ctx.textAlign = 'left'; ctx.textBaseline = 'middle';
    ctx.fillText('RALLY', 50, 95); ctx.fillText('RECORD', 50, 225);
    ctx.textAlign = 'right'; ctx.font = 'bold 110px -apple-system, Helvetica, Arial';
    ctx.fillStyle = key === 'hard' ? '#7fd1ff' : (key === 'grass' ? '#ffffff' : '#ffd166');
    ctx.fillText(String(streak), 974, 95);
    ctx.fillStyle = key === 'grass' ? '#c9a8ff' : '#7dff9a';
    ctx.fillText(String(record), 974, 225);
    ctx.fillStyle = 'rgba(255,255,255,0.25)'; ctx.fillRect(40, 158, 944, 4);
    tex.needsUpdate = true;
  };
  draw(0, 0);
  return draw;
}

// --- sky + reflected-light environment ---------------------------------------------------------
function skyMaterial() {
  return new THREE.ShaderMaterial({
    side: THREE.BackSide, depthWrite: false, fog: false,
    uniforms: { top: { value: new THREE.Color() }, bottom: { value: new THREE.Color() } },
    vertexShader: 'varying vec3 vW; void main(){ vW = (modelMatrix * vec4(position, 1.0)).xyz; gl_Position = projectionMatrix * viewMatrix * vec4(vW, 1.0); }',
    fragmentShader: 'uniform vec3 top; uniform vec3 bottom; varying vec3 vW; void main(){ float h = normalize(vW).y; gl_FragColor = vec4(mix(bottom, top, clamp(h * 2.2, 0.0, 1.0)), 1.0);\n#include <colorspace_fragment>\n}',
  });
}

function buildEnvironment(renderer, v, key) {
  // A tiny scene (sky + court color + bright lights) rendered into an environment
  // map, so shiny things (hard court, racket, rails) reflect the right colors.
  const env = new THREE.Scene();
  const sky = new THREE.Mesh(new THREE.SphereGeometry(50, 32, 16), skyMaterial());
  sky.material.uniforms.top.value.setHex(v.sky[0]);
  sky.material.uniforms.bottom.value.setHex(v.sky[1]);
  env.add(sky);
  const floor = new THREE.Mesh(new THREE.CircleGeometry(40, 32),
    new THREE.MeshBasicMaterial({ color: key === 'clay' ? 0xbf6038 : key === 'grass' ? 0x4f9444 : 0x3a679f }));
  floor.rotation.x = -Math.PI / 2; floor.position.y = -1; env.add(floor);
  const glow = new THREE.MeshBasicMaterial({ color: 0xffffff });
  glow.color.multiplyScalar(key === 'hard' ? 6 : 3);
  const sunBall = new THREE.Mesh(new THREE.SphereGeometry(3, 16, 8), glow);
  sunBall.position.set(...v.sun.pos).normalize().multiplyScalar(40); env.add(sunBall);
  const pmrem = new THREE.PMREMGenerator(renderer);
  const tex = pmrem.fromScene(env, 0.03).texture;
  pmrem.dispose();
  return tex;
}

function addStars(g) {
  const n = 600, pos = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    const a = Math.random() * Math.PI * 2, e = rnd(0.15, 1.3), r = 200;
    pos.set([Math.cos(a) * Math.cos(e) * r, Math.sin(e) * r, Math.sin(a) * Math.cos(e) * r], i * 3);
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.BufferAttribute(pos, 3));
  g.add(new THREE.Points(geo, new THREE.PointsMaterial({ color: 0xcfd8ff, size: 0.9, sizeAttenuation: true, fog: false, transparent: true, opacity: 0.7 })));
}

// --- public: build everything once, then switch venues ------------------------------------------
export function buildVenues(renderer, scene, { low = false } = {}) {
  ANISO = low ? 4 : renderer.capabilities.getMaxAnisotropy();
  QUALITY.crowd = low ? 0.4 : 0.9;

  const hemi = new THREE.HemisphereLight(0xffffff, 0x888888, 1);
  const sun = new THREE.DirectionalLight(0xffffff, 2.5);
  sun.target.position.set(0, 0, -12);
  sun.castShadow = true;
  sun.shadow.mapSize.setScalar(low ? 1024 : 2048);
  Object.assign(sun.shadow.camera, { left: -30, right: 30, top: 30, bottom: -30, near: 1, far: 120 });
  sun.shadow.bias = -0.0005;
  sun.shadow.normalBias = 0.02;
  const sky = new THREE.Mesh(new THREE.SphereGeometry(300, 32, 16), skyMaterial());
  scene.add(hemi, sun, sun.target, sky);
  scene.fog = new THREE.Fog(0xffffff, 80, 200);

  const built = {};
  for (const key of Object.keys(VENUES)) {
    const v = VENUES[key], g = new THREE.Group();
    buildSurface(g, key);
    buildWalls(g, v);
    let bowl;
    if (key === 'clay') {
      bowl = buildBowl(g, v, [{ rows: 26, rise: 0.42, depth: 0.82 }], { color: 0xbfc4c8, opacity: 1, reach: 4 });
    } else if (key === 'grass') {
      bowl = buildBowl(g, v, [{ rows: 30, rise: 0.5, depth: 0.8 }], { color: 0x6f767c, opacity: 0.92, reach: 8 });
    } else {
      bowl = buildBowl(g, v, [{ rows: 20, rise: 0.45, depth: 0.85 }, { band: 3.2 }, { rows: 26, rise: 0.66, depth: 0.9 }],
        { color: 0x1c2333, opacity: 1, reach: 11, lights: true });
      addStars(g);
    }
    buildCourtside(g, v);
    const drawBoard = buildBoard(g, v, key, bowl);
    g.visible = false;
    scene.add(g);
    built[key] = { group: g, env: buildEnvironment(renderer, v, key), drawBoard };
  }

  let current = null;
  return {
    apply(key) {
      const v = VENUES[key];
      for (const k in built) built[k].group.visible = k === key;
      sky.material.uniforms.top.value.setHex(v.sky[0]);
      sky.material.uniforms.bottom.value.setHex(v.sky[1]);
      scene.fog.color.setHex(v.fog[0]); scene.fog.near = v.fog[1]; scene.fog.far = v.fog[2];
      hemi.color.setHex(v.hemi[0]); hemi.groundColor.setHex(v.hemi[1]); hemi.intensity = v.hemi[2];
      sun.color.setHex(v.sun.color); sun.intensity = v.sun.intensity; sun.position.set(...v.sun.pos);
      renderer.toneMappingExposure = v.exposure;
      scene.environment = built[key].env;
      current = key;
      return v;
    },
    setScore(streak, record) {
      if (current) built[current].drawBoard(streak, record);
    },
  };
}
