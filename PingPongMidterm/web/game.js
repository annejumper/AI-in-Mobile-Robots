// T-anne-is - Three.js view (Anne Jumper, ME193)
//
// This page only DRAWS. tennis.py does all the sensing (pose, IMU, AprilTags),
// physics and decisions, and streams the game state here over a websocket.
// Positions arrive in the game's world meters: x right, y up, z toward the
// opponent. Three.js looks down -z, so z is flipped when placing things.

import * as THREE from 'three';

import { EffectComposer } from './vendor/jsm/postprocessing/EffectComposer.js';
import { RenderPass } from './vendor/jsm/postprocessing/RenderPass.js';
import { UnrealBloomPass } from './vendor/jsm/postprocessing/UnrealBloomPass.js';
import { OutputPass } from './vendor/jsm/postprocessing/OutputPass.js';
import {
  V, canvasTex, addNoise, buildVenues, VENUES,
  BASELINE_Z, NET_Z, FAR_Z, OPP_Z, NET_H, POST_H, POST_X,
} from './venues.js';

const BALL_R = 0.06;  // drawn ~1.8x real size so it's easy to track
const LOW = new URLSearchParams(location.search).has('low');   // add &low to the URL on a slow laptop

// --- renderer, camera, post-processing ----------------------------------------------
const renderer = new THREE.WebGLRenderer({ canvas: document.getElementById('scene'), antialias: true });
renderer.setPixelRatio(LOW ? 1 : Math.min(window.devicePixelRatio, 2));
renderer.shadowMap.enabled = true;
renderer.shadowMap.type = THREE.PCFSoftShadowMap;
renderer.toneMapping = THREE.ACESFilmicToneMapping;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(52, 1, 0.05, 500);
const CAM_BASE = new THREE.Vector3(0, 2.45, 4.6);   // just behind and above the player
const LOOK_AT = new THREE.Vector3(0, 0.55, -14);

// bloom: bright things (floodlights, LED board, white lines at night) glow softly
const composer = new EffectComposer(renderer);
composer.addPass(new RenderPass(scene, camera));
const bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.3, 0.6, 0.85);
bloom.enabled = !LOW;
composer.addPass(bloom);
composer.addPass(new OutputPass());

function resize() {
  renderer.setSize(window.innerWidth, window.innerHeight, false);
  composer.setPixelRatio(renderer.getPixelRatio());
  composer.setSize(window.innerWidth, window.innerHeight);
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
}
window.addEventListener('resize', resize);
resize();

// the three stadiums (venues.js): court, walls, 3D crowd, roof, lights, sky
const venues = buildVenues(renderer, scene, { low: LOW });

const blobTex = canvasTex(128, 128, (c, w, h) => {
  const g = c.createRadialGradient(w / 2, h / 2, 0, w / 2, h / 2, w / 2);
  g.addColorStop(0, 'rgba(0,0,0,0.65)'); g.addColorStop(1, 'rgba(0,0,0,0)');
  c.fillStyle = g; c.fillRect(0, 0, w, h);
});

function buildNet() {
  const g = new THREE.Group(), width = 2 * POST_X;
  const sag = (x, top) => top ? NET_H + (POST_H - NET_H) * (x / (width / 2)) ** 2 : 0;
  const netGeo = new THREE.PlaneGeometry(width, 1, 64, 1);
  const tapeGeo = new THREE.PlaneGeometry(width, 1, 64, 1);
  for (const [geo, tape] of [[netGeo, false], [tapeGeo, true]]) {
    const p = geo.attributes.position;
    for (let i = 0; i < p.count; i++) {
      const top = p.getY(i) > 0, h = sag(p.getX(i), true);
      p.setY(i, tape ? (top ? h + 0.01 : h - 0.06) : (top ? h : 0));
    }
    geo.computeVertexNormals();
  }
  const meshTex = canvasTex(32, 32, (c, w, h) => {
    c.clearRect(0, 0, w, h); c.strokeStyle = 'rgba(25,25,25,1)'; c.lineWidth = 4; c.strokeRect(0, 0, w, h);
  }, [width / 0.045, 1 / 0.045]);
  g.add(new THREE.Mesh(netGeo, new THREE.MeshStandardMaterial({ map: meshTex, transparent: true, alphaTest: 0.2, side: THREE.DoubleSide, roughness: 1 })));
  g.add(new THREE.Mesh(tapeGeo, new THREE.MeshStandardMaterial({ color: 0xffffff, side: THREE.DoubleSide, roughness: 0.6 })));
  const postMat = new THREE.MeshStandardMaterial({ color: 0x1b3326, roughness: 0.4, metalness: 0.4 });
  for (const sx of [-1, 1]) {
    const post = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.04, POST_H + 0.03, 12), postMat);
    post.position.set(sx * POST_X, (POST_H + 0.03) / 2, 0); post.castShadow = true; g.add(post);
  }
  const strap = new THREE.Mesh(new THREE.BoxGeometry(0.05, NET_H, 0.012), new THREE.MeshStandardMaterial({ color: 0xffffff }));
  strap.position.set(0, NET_H / 2, 0); g.add(strap);
  g.position.copy(V(0, 0, NET_Z));
  scene.add(g);
  return postMat;
}
const netPostMat = buildNet();

// --- racket, players, ball ---------------------------------------------------------------
const stringTex = canvasTex(128, 160, (c, w, h) => {
  c.clearRect(0, 0, w, h); c.strokeStyle = 'rgba(245,245,235,0.95)'; c.lineWidth = 2;
  for (let x = 4; x < w; x += 9) { c.beginPath(); c.moveTo(x, 0); c.lineTo(x, h); c.stroke(); }
  for (let y = 4; y < h; y += 9) { c.beginPath(); c.moveTo(0, y); c.lineTo(w, y); c.stroke(); }
});

function makeRacket(color, scale = 1) {
  const outer = new THREE.Group(), inner = new THREE.Group();
  const frameMat = new THREE.MeshStandardMaterial({ color, roughness: 0.35, metalness: 0.3, emissive: 0x000000 });
  const head = new THREE.Mesh(new THREE.TorusGeometry(0.15, 0.012, 10, 48), frameMat);
  head.scale.set(0.8, 1, 1);
  const strings = new THREE.Mesh(new THREE.PlaneGeometry(0.235, 0.295),
    new THREE.MeshStandardMaterial({ map: stringTex, transparent: true, alphaTest: 0.2, side: THREE.DoubleSide }));
  const throat = new THREE.Mesh(new THREE.CylinderGeometry(0.011, 0.014, 0.12, 10), frameMat);
  throat.position.y = -0.2;
  const handle = new THREE.Mesh(new THREE.CylinderGeometry(0.017, 0.017, 0.2, 10),
    new THREE.MeshStandardMaterial({ color: 0x151515, roughness: 0.9 }));
  handle.position.y = -0.36;
  for (const m of [head, throat, handle]) m.castShadow = true;
  inner.add(head, strings, throat, handle);
  inner.scale.setScalar(scale);
  outer.add(inner);
  outer.userData = { inner, frameMat, swingT: 1 };
  return outer;
}

const racket = makeRacket(0xff3d7f);
racket.visible = false;
scene.add(racket);

function makeOpponent() {
  const g = new THREE.Group();
  const skin = new THREE.MeshStandardMaterial({ color: 0xd9a77f, roughness: 0.8 });
  const shirt = new THREE.MeshStandardMaterial({ color: 0xf2f2f2, roughness: 0.8 });
  const shorts = new THREE.MeshStandardMaterial({ color: 0x1d2433, roughness: 0.8 });
  for (const sx of [-0.11, 0.11]) {
    const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.065, 0.06, 0.85, 10), skin);
    leg.position.set(sx, 0.43, 0); g.add(leg);
  }
  const hips = new THREE.Mesh(new THREE.CylinderGeometry(0.2, 0.19, 0.25, 14), shorts);
  hips.position.y = 0.92; g.add(hips);
  const torso = new THREE.Mesh(new THREE.CapsuleGeometry(0.2, 0.42, 6, 14), shirt);
  torso.position.y = 1.3; g.add(torso);
  const head = new THREE.Mesh(new THREE.SphereGeometry(0.12, 20, 14), skin);
  head.position.y = 1.78; g.add(head);
  const cap = new THREE.Mesh(new THREE.SphereGeometry(0.125, 20, 10, 0, Math.PI * 2, 0, Math.PI / 2), shirt);
  cap.position.y = 1.8; g.add(cap);
  const r = makeRacket(0x2a2a2a);
  r.position.set(0.42, 1.15, 0.05);
  r.rotation.z = 0.5;
  g.add(r);
  g.traverse(o => { if (o.isMesh) o.castShadow = true; });
  g.userData = { racket: r, x: 0, shirt };
  g.position.copy(V(0, 0, OPP_Z + 0.4));
  scene.add(g);
  return g;
}
const opponent = makeOpponent();

const ballTex = canvasTex(256, 128, (c, w, h) => {
  c.fillStyle = '#d6f23e'; c.fillRect(0, 0, w, h); addNoise(c, w, h, 14);
  c.strokeStyle = '#f4f7ea'; c.lineWidth = 7; c.beginPath();
  for (let x = 0; x <= w; x++) {
    const y = h / 2 + Math.sin((x / w) * Math.PI * 4) * h * 0.27;
    if (x) c.lineTo(x, y); else c.moveTo(x, y);
  }
  c.stroke();
});
const ball = new THREE.Mesh(new THREE.SphereGeometry(BALL_R, 32, 16),
  new THREE.MeshStandardMaterial({ map: ballTex, roughness: 0.9, emissive: 0x2c3a00, emissiveIntensity: 0.7 }));
ball.visible = false;
scene.add(ball);

const blob = new THREE.Mesh(new THREE.PlaneGeometry(1, 1),
  new THREE.MeshBasicMaterial({ map: blobTex, transparent: true, depthWrite: false }));
blob.rotation.x = -Math.PI / 2;
blob.visible = false;
scene.add(blob);

const TRAIL = 14;
const trail = [];
for (let i = 0; i < TRAIL; i++) {
  const m = new THREE.Mesh(new THREE.SphereGeometry(0.03, 8, 6),
    new THREE.MeshBasicMaterial({ color: 0xeaff70, transparent: true, opacity: 0, depthWrite: false }));
  scene.add(m); trail.push(m);
}
const trailPts = [];

// "Hit here" ring at the predicted contact point (on the player's contact plane, z = 0)
const ring = new THREE.Group();
const ringMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.85, toneMapped: false });
ring.add(new THREE.Mesh(new THREE.TorusGeometry(1, 0.035, 12, 72), ringMat));
const ringDisc = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.07, depthWrite: false, toneMapped: false });
ring.add(new THREE.Mesh(new THREE.CircleGeometry(1, 48), ringDisc));
ring.visible = false;
scene.add(ring);

// --- effects: dust, ball marks, landing pulses ---------------------------------------------
const MAXP = 200;
const pGeo = new THREE.BufferGeometry();
const pPos = new Float32Array(MAXP * 3).fill(-50), pCol = new Float32Array(MAXP * 3);
pGeo.setAttribute('position', new THREE.BufferAttribute(pPos, 3));
pGeo.setAttribute('color', new THREE.BufferAttribute(pCol, 3));
const particles = Array.from({ length: MAXP }, () => ({ v: new THREE.Vector3(), life: 0 }));
const points = new THREE.Points(pGeo, new THREE.PointsMaterial({ size: 0.07, vertexColors: true, transparent: true, opacity: 0.85, depthWrite: false }));
points.frustumCulled = false;
scene.add(points);
let pNext = 0;

function dust(pos, rgb, n, speed = 1.6) {
  for (let k = 0; k < n; k++) {
    const i = pNext++ % MAXP, p = particles[i];
    const a = Math.random() * Math.PI * 2, s = speed * (0.3 + Math.random());
    p.v.set(Math.cos(a) * s, 0.6 + Math.random() * 1.4, Math.sin(a) * s);
    p.life = 0.5 + Math.random() * 0.5;
    pPos.set([pos.x, 0.03, pos.z], i * 3);
    const j = 0.85 + Math.random() * 0.3;
    pCol.set([rgb[0] * j, rgb[1] * j, rgb[2] * j], i * 3);
  }
  pGeo.attributes.color.needsUpdate = true;
}

function updateParticles(dt) {
  for (let i = 0; i < MAXP; i++) {
    const p = particles[i];
    if (p.life <= 0) continue;
    p.life -= dt;
    p.v.y -= 3.0 * dt;
    p.v.multiplyScalar(0.96);
    pPos[i * 3] += p.v.x * dt; pPos[i * 3 + 1] += p.v.y * dt; pPos[i * 3 + 2] += p.v.z * dt;
    if (p.life <= 0 || pPos[i * 3 + 1] < 0) { p.life = 0; pPos[i * 3 + 1] = -50; }
  }
  pGeo.attributes.position.needsUpdate = true;
}

const marks = [];
function ballMark(pos, color) {
  const m = new THREE.Mesh(new THREE.CircleGeometry(0.06, 20),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.55, depthWrite: false }));
  m.rotation.x = -Math.PI / 2; m.scale.set(1, 1.9, 1);
  m.position.set(pos.x, 0.006, pos.z);
  scene.add(m); marks.push(m);
  if (marks.length > 10) scene.remove(marks.shift());
}

const pulses = [];
function pulse(pos, color) {
  const m = new THREE.Mesh(new THREE.RingGeometry(0.18, 0.26, 40),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0.9, depthWrite: false, toneMapped: false, side: THREE.DoubleSide }));
  m.rotation.x = -Math.PI / 2; m.position.set(pos.x, 0.01, pos.z);
  m.userData.t = 0; scene.add(m); pulses.push(m);
}

function updatePulses(dt) {
  for (let i = pulses.length - 1; i >= 0; i--) {
    const m = pulses[i];
    m.userData.t += dt;
    const k = m.userData.t / 0.7;
    m.scale.setScalar(1 + k * 3); m.material.opacity = 0.9 * (1 - k);
    if (k >= 1) { scene.remove(m); pulses.splice(i, 1); }
  }
}

// --- sound (Web Audio, synthesized: no files needed) ----------------------------------------
let actx = null, noiseBuf = null;
function audio() {
  if (!actx) {
    actx = new (window.AudioContext || window.webkitAudioContext)();
    noiseBuf = actx.createBuffer(1, actx.sampleRate, actx.sampleRate);
    const d = noiseBuf.getChannelData(0);
    for (let i = 0; i < d.length; i++) d[i] = Math.random() * 2 - 1;
  }
  if (actx.state === 'suspended') actx.resume();
  document.getElementById('soundHint').classList.add('hidden');
}
window.addEventListener('pointerdown', audio);

const live = () => actx && actx.state === 'running';

function thock({ freq = 180, dur = 0.09, gain = 0.4, noise = 0.6, nfreq = 1800 }) {
  if (!live()) return;
  const t = actx.currentTime;
  const o = actx.createOscillator(), g = actx.createGain();
  o.type = 'triangle';
  o.frequency.setValueAtTime(freq * 1.7, t);
  o.frequency.exponentialRampToValueAtTime(freq, t + dur);
  g.gain.setValueAtTime(gain, t); g.gain.exponentialRampToValueAtTime(0.001, t + dur);
  o.connect(g).connect(actx.destination); o.start(t); o.stop(t + dur + 0.02);
  const s = actx.createBufferSource(), f = actx.createBiquadFilter(), g2 = actx.createGain();
  s.buffer = noiseBuf; f.type = 'bandpass'; f.frequency.value = nfreq; f.Q.value = 1.4;
  g2.gain.setValueAtTime(gain * noise, t); g2.gain.exponentialRampToValueAtTime(0.001, t + dur * 0.6);
  s.connect(f).connect(g2).connect(actx.destination); s.start(t); s.stop(t + dur);
}

function tone(freq, dur, gain = 0.15, type = 'sine', delay = 0, slideTo = null) {
  if (!live()) return;
  const t = actx.currentTime + delay, o = actx.createOscillator(), g = actx.createGain();
  o.type = type; o.frequency.setValueAtTime(freq, t);
  if (slideTo) o.frequency.exponentialRampToValueAtTime(slideTo, t + dur);
  g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(gain, t + 0.01);
  g.gain.exponentialRampToValueAtTime(0.001, t + dur);
  o.connect(g).connect(actx.destination); o.start(t); o.stop(t + dur + 0.05);
}

function applause() {
  if (!live()) return;
  const t = actx.currentTime, s = actx.createBufferSource(), f = actx.createBiquadFilter(), g = actx.createGain();
  s.buffer = noiseBuf; s.loop = true; f.type = 'bandpass'; f.frequency.value = 2200; f.Q.value = 0.6;
  g.gain.setValueAtTime(0.0001, t); g.gain.exponentialRampToValueAtTime(0.22, t + 0.35);
  g.gain.exponentialRampToValueAtTime(0.001, t + 2.6);
  s.connect(f).connect(g).connect(actx.destination); s.start(t); s.stop(t + 2.7);
}

// --- HUD -------------------------------------------------------------------------------------
const $ = id => document.getElementById(id);
const setText = (el, t) => { if (el.textContent !== String(t)) el.textContent = t; };
const show = (el, on) => el.classList.toggle('hidden', !on);

let S = null, recvAt = 0, surface = null, lastMsgTitle = null, lastRecord = 0, lastStreak = 0;

function setSurface(key) {
  surface = key;
  const v = venues.apply(key);
  bloom.strength = v.bloom;
  netPostMat.color.setHex(v.post);
  opponent.userData.shirt.color.setHex(v.shirt);
  venues.setScore(S ? S.streak : 0, S ? S.record : 0);
  document.documentElement.style.setProperty('--accent', v.accent);
  for (const m of marks) scene.remove(m);
  marks.length = 0;
}

function bump(el) { el.classList.remove('bump'); void el.offsetWidth; el.classList.add('bump'); }

function updateHUD(s) {
  setText($('surfaceName'), `${VENUES[s.surface].title} \u00b7 ${s.surfaceName.toLowerCase()} court`);
  setText($('blurb'), s.blurb);
  setText($('levelLine'), `Level ${s.level}: ${s.levelName} · ${s.speed} m/s`);
  setText($('streak'), s.streak);
  setText($('record'), s.record);
  if (s.streak > lastStreak) bump($('streak'));
  if (s.record > lastRecord) bump($('record'));
  if (s.streak !== lastStreak || s.record !== lastRecord) venues.setScore(s.streak, s.record);
  lastStreak = s.streak; lastRecord = s.record;
  setText($('topicText'), s.topic);
  $('mqttDot').classList.toggle('on', s.mqtt);

  // IMU meters
  setText($('imuTitle'), s.sim ? 'IMU · KEYBOARD SIM' : 'IMU · DOUBLE MOTOR 6065');
  const sw = Math.min(1, s.imu.swing);
  $('swingFill').style.width = `${sw * 100}%`;
  $('swingFill').classList.toggle('hot', sw >= s.imu.swingTick);
  $('swingTick').style.left = `${s.imu.swingTick * 100}%`;
  const tw = Math.max(-1, Math.min(1, s.imu.twist)), dead = s.imu.twistDead;
  $('deadBand').style.left = `${50 - dead * 50}%`; $('deadBand').style.width = `${dead * 100}%`;
  const tf = $('twistFill');
  tf.style.left = `${tw >= 0 ? 50 : 50 + tw * 50}%`; tf.style.width = `${Math.abs(tw) * 50}%`;
  tf.style.background = Math.abs(tw) < dead ? 'rgba(255,255,255,0.5)' : (tw > 0 ? 'var(--topspin)' : 'var(--slice)');
  if (s.lastSwing) {
    const k = s.lastSwing.spin > 0.15 ? 'topspin' : s.lastSwing.spin < -0.15 ? 'slice' : 'flat';
    setText($('lastSwing'), `last swing: power ${Math.round(s.lastSwing.power * 100)}% · ${k} ${s.lastSwing.spin >= 0 ? '+' : ''}${s.lastSwing.spin.toFixed(2)}` +
      (s.lastSwing.label ? ` · KNN: ${s.lastSwing.label.replace('_', ' ')}` : ''));
  }

  // last shot card
  const shot = s.lastShot;
  show($('shot'), !!shot);
  if (shot) {
    $('shotMain').innerHTML = `${shot.stroke}<span class="spinTag spin-${shot.label}">${shot.label.toLowerCase()}</span>`;
    show($('perfect'), shot.perfect);
    setText($('sSpeed'), `${Math.round(shot.speed)} m/s`);
    setText($('sPower'), `${Math.round(shot.power * 100)}%`);
    setText($('sSpin'), `${shot.spin >= 0 ? '+' : ''}${shot.spin.toFixed(2)}`);
    setText($('sHeight'), `${Math.round(shot.height * 100)} cm`);
  }

  // menu
  show($('menu'), s.state === 'MENU');
  $('startFill').style.width = `${s.startSeen * 100}%`;
  document.querySelectorAll('.tag').forEach(el => {
    el.classList.toggle('active', el.dataset.level == s.level || el.dataset.surface === s.surface);
  });
  show($('simKeys'), s.sim);

  // countdown, banner, tracking warning
  show($('countdown'), s.state === 'COUNTDOWN');
  if (s.state === 'COUNTDOWN') setText($('countdown'), s.countdown > 0 ? s.countdown : 'PLAY!');
  const b = $('banner');
  show(b, !!s.msg);
  if (s.msg) {
    b.classList.remove('good', 'bad', 'warn', 'info');
    b.classList.add(s.msg.kind);
    setText($('bTitle'), s.msg.title); setText($('bDetail'), s.msg.detail);
    if (s.msg.title !== lastMsgTitle) { b.classList.remove('pop'); void b.offsetWidth; b.classList.add('pop'); }
  }
  lastMsgTitle = s.msg ? s.msg.title : null;
  show($('noTrack'), !s.tracking);
}

// --- Chris commentary: a reaction face + one-liner for each shot -----------------------------
const CHRIS_LINES = {
  happy:   ['Lovely shot!', 'That\'s in. Keep it going!', 'Clean contact.', 'Textbook.',
            'You\'re a natural. Have you considered frisbee?',
            'Great hands! Frisbee\'s going to love you.',
            'With aim like that, you\'d be a frisbee star.',
            'Solid! Now try it with a disc.'],
  shocked: ['WHAT a shot!', 'Did you see that?!', 'Unbelievable!', 'Somebody call Wimbledon!',
            'Talent like that is wasted on tennis. Ultimate needs you!',
            'Wow! Somebody sign this player up for frisbee!',
            'That\'s frisbee-level talent right there!',
            'Wow, you may graduate on time after all!'],
  sad:     ['Oof. So close.', 'Ahh, not quite.', 'You\'ll get the next one.', 'Painful to watch.',
            'At least a frisbee floats long enough to catch.',
            'A frisbee would\'ve waited for you.',
            'That ball was tiny anyway. A disc is much easier to see.',
            'Don\'t worry, frisbee doesn\'t need a racket.',
            'Tough one. Frisbee is always there for you.'],
  mad:     ['Come ON!', 'Are you serious?!', 'That was a sitter!', 'Unacceptable.',
            'This is why I play a superior sport. FRISBEE.',
            'That never happens with a frisbee!',
            'Into the net? A frisbee would\'ve flown right over!',
            'Rackets are overrated. Just throw a frisbee!'],
};
const pick = a => a[Math.floor(Math.random() * a.length)];
const chrisFaces = Object.keys(CHRIS_LINES).map(m => Object.assign(new Image(), { src: `img/chris-${m}.png` }));  // preload

function chrisSays(mood, line) {
  const panel = $('chris'), img = $('chrisImg');
  panel.classList.remove('happy', 'shocked', 'sad', 'mad');
  panel.classList.add(mood);
  img.src = `img/chris-${mood}.png`;
  setText($('chrisQuote'), line || pick(CHRIS_LINES[mood]));
  img.classList.remove('react'); void img.offsetWidth; img.classList.add('react');
}

function chrisReact(e) {
  switch (e.type) {
    case 'hit':
      if (e.perfect) chrisSays('shocked', 'PERFECT timing!');
      else if (e.power >= 0.85) chrisSays('shocked');
      break;
    case 'in':
      if (S && S.lastShot && !S.lastShot.perfect && S.lastShot.power < 0.85) chrisSays('happy');
      break;
    case 'record': chrisSays('shocked', 'A new record!'); break;
    case 'miss':   // unforced errors (net / long) make him mad; whiffs make him sad
      chrisSays(/net|long/i.test(e.title || '') ? 'mad' : 'sad');
      break;
  }
}

// --- events from the game (sounds + effects) -----------------------------------------------
let shake = 0;
function handleFx(fx) {
  for (const e of fx) {
    chrisReact(e);
    const p = e.pos ? V(e.pos[0], e.pos[1], e.pos[2]) : null;
    switch (e.type) {
      case 'bounce': {
        const sf = VENUES[surface].fx;
        const near = Math.max(0.25, 1 - e.pos[2] / 40);
        thock({ ...sf.bounce, gain: 0.32 * near, dur: 0.08 });
        if (sf.dust) dust(p, sf.dust, sf.dustN);
        if (sf.mark) ballMark(p, sf.mark);
        if (e.pos[2] > FAR_Z + 0.05) pulse(p, 0xff5d5d);       // long!
        break;
      }
      case 'hit':
        thock({ freq: 240, gain: 0.35 + 0.45 * e.power, nfreq: 2600 });
        if (e.perfect) tone(1568, 0.14, 0.12, 'sine', 0.03);
        racket.userData.swingT = 0;
        shake = 0.02 + 0.04 * e.power;
        break;
      case 'oppHit':
        thock({ freq: 230, gain: 0.14, nfreq: 2300 });
        opponent.userData.racket.userData.swingT = 0;
        break;
      case 'in': pulse(p, 0x5fdc7a); break;
      case 'net': thock({ freq: 80, gain: 0.25, noise: 1, nfreq: 300, dur: 0.15 }); break;
      case 'miss': tone(330, 0.4, 0.14, 'triangle', 0, 110); break;
      case 'record':
        [523, 659, 784, 1047].forEach((f, i) => tone(f, 0.18, 0.13, 'triangle', i * 0.1));
        applause();
        break;
      case 'count': tone(e.n ? 660 : 1320, 0.15, 0.13); break;
    }
  }
}

// --- connection to tennis.py -------------------------------------------------------------------
const WS_PORT = new URLSearchParams(location.search).get('ws') || '8766';
let ws = null, camUrl = null;

function connect() {
  ws = new WebSocket(`ws://${location.hostname}:${WS_PORT}`);
  ws.binaryType = 'blob';
  ws.onopen = () => show($('offline'), false);
  ws.onclose = () => { show($('offline'), true); setTimeout(connect, 1000); };
  ws.onmessage = ev => {
    if (typeof ev.data === 'string') {
      const s = JSON.parse(ev.data);
      if (s.surface !== surface) setSurface(s.surface);
      S = s; recvAt = performance.now();
      handleFx(s.fx);
      updateHUD(s);
    } else {
      const url = URL.createObjectURL(ev.data), old = camUrl;
      camUrl = url;
      $('camImg').src = url;
      if (old) setTimeout(() => URL.revokeObjectURL(old), 500);
    }
  };
}
connect();

const KEYS = new Set(['t', 'b', 'f', '1', '2', '3', 'c', 'g', 'h', 's', 'q']);
window.addEventListener('keydown', e => {
  audio();
  const k = e.key === ' ' ? 'space' : e.key.toLowerCase();
  if (k === 'space') e.preventDefault();
  if ((k === 'space' || KEYS.has(k)) && ws && ws.readyState === 1) ws.send(JSON.stringify({ key: k }));
});

// --- per-frame animation -------------------------------------------------------------------------
const clock = new THREE.Clock();
const tmp = new THREE.Vector3();

function animate() {
  requestAnimationFrame(animate);
  const dt = Math.min(clock.getDelta(), 0.05), t = clock.elapsedTime;
  const s = S;

  // ball: extrapolate between 30 Hz updates so it moves smoothly at 60 fps
  if (s && s.ball) {
    const lag = Math.min((performance.now() - recvAt) / 1000, 0.05);
    const [x, y, z] = s.ball.pos, [vx, vy, vz] = s.ball.vel;
    const bx = x + vx * lag, by = Math.max(0, y + vy * lag), bz = z + vz * lag;
    ball.position.copy(V(bx, by + BALL_R * 0.5, bz));
    ball.rotation.x += dt * (8 + 30 * Math.abs(s.ball.spin)) * Math.sign(vz || 1) * (s.ball.spin < 0 ? -1 : 1);
    ball.visible = blob.visible = true;
    blob.position.copy(V(bx, 0.008, bz));
    blob.scale.setScalar(0.22 + by * 0.12);
    blob.material.opacity = 0.9 / (1 + by * 0.8);
    trailPts.push(ball.position.clone());
    if (trailPts.length > TRAIL) trailPts.shift();
  } else {
    ball.visible = blob.visible = false;
    trailPts.length = 0;
  }
  trail.forEach((m, i) => {
    const p = trailPts[trailPts.length - 1 - i - 1];
    m.visible = !!p;
    if (p) { m.position.copy(p); m.material.opacity = 0.4 * (1 - i / TRAIL); m.scale.setScalar(1 - i / TRAIL); }
  });

  // racket follows the player's hand (from MediaPipe pose), on the contact plane z = 0
  const rd = racket.userData;
  if (s && s.racket && s.tracking) {
    tmp.copy(V(s.racket[0], s.racket[1], 0));
    if (!racket.visible) racket.position.copy(tmp);
    racket.position.lerp(tmp, 0.5);
    racket.visible = true;
    // handle points back toward the player's body
    const dx = s.racket[0] - 0, dy = s.racket[1] - 0.85;
    const len = Math.hypot(dx, dy) || 1;
    racket.rotation.z = Math.atan2(-dx / len, dy / len) * 0.9;
    rd.frameMat.emissive.setHex(s.swinging ? 0x661020 : 0x000000);
  } else {
    racket.visible = false;
  }
  for (const r of [racket, opponent.userData.racket]) {
    const u = r.userData;
    u.swingT = Math.min(1, u.swingT + dt / 0.25);
    const k = Math.sin(u.swingT * Math.PI);           // 0 -> 1 -> 0
    u.inner.rotation.y = -k * 1.1;
    u.inner.rotation.x = -k * 0.4;
  }

  // ring: where the ball will be; turns green when your racket is in it
  if (s && s.ghost) {
    ring.visible = true;
    ring.position.copy(V(s.ghost[0], s.ghost[1], 0));
    ring.scale.setScalar(s.hitRadius);
    const inside = s.racket && Math.hypot(s.racket[0] - s.ghost[0], s.racket[1] - s.ghost[1]) <= s.hitRadius;
    ringMat.color.setHex(inside ? 0x5fdc7a : 0xffffff);
    ringDisc.color.copy(ringMat.color);
    ringMat.opacity = 0.6 + 0.3 * Math.sin(t * 8);
  } else {
    ring.visible = false;
  }

  // opponent shuffles toward the ball
  const ou = opponent.userData;
  const target = s && s.ball && !s.ball.incoming ? Math.max(-6, Math.min(6, s.ball.pos[0])) : 0;
  ou.x += Math.max(-5 * dt, Math.min(5 * dt, target - ou.x));
  opponent.position.x = ou.x;
  opponent.position.y = Math.abs(Math.sin(t * 6)) * 0.03;

  updateParticles(dt);
  updatePulses(dt);

  // camera: steady, a touch of parallax with your racket, a kick on big hits
  shake *= Math.pow(0.002, dt);
  camera.position.set(
    CAM_BASE.x + (racket.visible ? racket.position.x * 0.12 : 0) + (Math.random() - 0.5) * shake,
    CAM_BASE.y + (Math.random() - 0.5) * shake,
    CAM_BASE.z);
  camera.lookAt(LOOK_AT);

  composer.render(dt);
}
setSurface('hard');
animate();
