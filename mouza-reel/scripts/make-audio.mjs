// Generates every non-voice sound in the reel FROM CODE (no stock, no samples):
//   public/sfx/pop.wav, tap.wav, ding.wav, whoosh.wav, roomtone.wav
//   public/music/bed.wav  (27 s lo-fi piano bed, drums enter at 13.0 s = scene 5)
// Run:  npm run sfx
// Swap any file for your own later – keep the same file names.

import {writeFileSync, mkdirSync} from 'node:fs';
import {dirname, join} from 'node:path';
import {fileURLToPath} from 'node:url';

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const SR = 48000;

// deterministic PRNG so every run produces identical files
let seed = 1337;
const rnd = () => {
  seed = (seed * 1664525 + 1013904223) >>> 0;
  return seed / 4294967296;
};

function writeWav(relPath, left, right = left) {
  const n = left.length;
  const buf = Buffer.alloc(44 + n * 4);
  buf.write('RIFF', 0);
  buf.writeUInt32LE(36 + n * 4, 4);
  buf.write('WAVE', 8);
  buf.write('fmt ', 12);
  buf.writeUInt32LE(16, 16);
  buf.writeUInt16LE(1, 20); // PCM
  buf.writeUInt16LE(2, 22); // stereo
  buf.writeUInt32LE(SR, 24);
  buf.writeUInt32LE(SR * 4, 28);
  buf.writeUInt16LE(4, 32);
  buf.writeUInt16LE(16, 34);
  buf.write('data', 36);
  buf.writeUInt32LE(n * 4, 40);
  for (let i = 0; i < n; i++) {
    const l = Math.max(-1, Math.min(1, left[i]));
    const r = Math.max(-1, Math.min(1, right[i]));
    buf.writeInt16LE(Math.round(l * 32767), 44 + i * 4);
    buf.writeInt16LE(Math.round(r * 32767), 46 + i * 4);
  }
  const out = join(ROOT, relPath);
  mkdirSync(dirname(out), {recursive: true});
  writeFileSync(out, buf);
  console.log('wrote', relPath, (n / SR).toFixed(2) + 's');
}

function normalize(arrs, peakDb = -3) {
  let peak = 0;
  for (const a of arrs) for (const v of a) peak = Math.max(peak, Math.abs(v));
  const target = Math.pow(10, peakDb / 20);
  const g = peak > 0 ? target / peak : 1;
  for (const a of arrs) for (let i = 0; i < a.length; i++) a[i] *= g;
}

const buf = (sec) => new Float32Array(Math.ceil(sec * SR));

// ---------- SFX ----------
function pop() {
  const b = buf(0.14);
  let ph = 0;
  for (let i = 0; i < b.length; i++) {
    const t = i / SR;
    const f = 180 + 760 * Math.exp(-t * 45);
    ph += (2 * Math.PI * f) / SR;
    b[i] = Math.sin(ph) * Math.exp(-t * 32) * Math.min(1, t * 2000);
  }
  normalize([b], -4);
  return b;
}

function tap() {
  const b = buf(0.06);
  let lp = 0;
  for (let i = 0; i < b.length; i++) {
    const t = i / SR;
    const n = rnd() * 2 - 1;
    lp += 0.35 * (n - lp);
    b[i] = (lp * 0.7 + Math.sin(2 * Math.PI * 1900 * t) * 0.3) * Math.exp(-t * 140);
  }
  normalize([b], -8);
  return b;
}

function ding() {
  const b = buf(1.2);
  const partials = [
    [1318.5, 1],
    [2637, 0.35],
    [3955, 0.12],
    [1975.5, 0.25],
  ];
  for (let i = 0; i < b.length; i++) {
    const t = i / SR;
    let v = 0;
    for (const [f, a] of partials) v += a * Math.sin(2 * Math.PI * f * t) * Math.exp(-t * (3 + f / 900));
    b[i] = v * Math.min(1, t * 800);
  }
  normalize([b], -6);
  return b;
}

function whoosh() {
  const b = buf(0.35);
  let lp = 0;
  let lp2 = 0;
  for (let i = 0; i < b.length; i++) {
    const t = i / SR;
    const k = 0.02 + 0.25 * Math.sin((Math.PI * t) / 0.35);
    lp += k * (rnd() * 2 - 1 - lp);
    lp2 += k * (lp - lp2);
    b[i] = (lp - lp2) * Math.sin((Math.PI * t) / 0.35) ** 2;
  }
  normalize([b], -8);
  return b;
}

function roomtone() {
  // phone-mic room tone: brown noise + faint 50 Hz hum (Israeli mains) – very quiet
  const l = buf(27);
  const r = buf(27);
  let bl = 0;
  let br = 0;
  for (let i = 0; i < l.length; i++) {
    const t = i / SR;
    bl = (bl + 0.02 * (rnd() * 2 - 1)) * 0.998;
    br = (br + 0.02 * (rnd() * 2 - 1)) * 0.998;
    const hum = 0.004 * Math.sin(2 * Math.PI * 50 * t);
    l[i] = bl + hum;
    r[i] = br + hum;
  }
  normalize([l, r], -30);
  return [l, r];
}

// ---------- MUSIC ----------
const BPM = 92;
const BEAT = 60 / BPM;
const BAR = BEAT * 4;
const DRUMS_IN = 13.0; // scene 5 starts at 13.0 s
const LEN = 27;
const midi = (m) => 440 * Math.pow(2, (m - 69) / 12);

// warm lo-fi progression: Am9 – Fmaj7 – Cmaj7 – G6
const CHORDS = [
  {bass: 45, notes: [60, 64, 67, 71]},
  {bass: 41, notes: [57, 60, 64, 67]},
  {bass: 48, notes: [59, 64, 67, 71]},
  {bass: 43, notes: [59, 62, 64, 67]},
];
const MELODY = [76, 74, 72, 69, 72, 74, 76, 79, 76, 74, 72, 74];

function addPianoNote(L, R, start, m, vel, dur = 1.6, pan = 0) {
  const f = midi(m) * (1 + (rnd() - 0.5) * 0.002); // slight detune = lo-fi
  const s0 = Math.floor(start * SR);
  const n = Math.floor(dur * SR);
  let lp = 0;
  for (let i = 0; i < n && s0 + i < L.length; i++) {
    const t = i / SR;
    const env = Math.min(1, t * 300) * Math.exp(-t * 2.4);
    let v =
      Math.sin(2 * Math.PI * f * t) +
      0.45 * Math.sin(2 * Math.PI * 2 * f * t) * Math.exp(-t * 3) +
      0.2 * Math.sin(2 * Math.PI * 3 * f * t) * Math.exp(-t * 5) +
      0.08 * Math.sin(2 * Math.PI * 4 * f * t) * Math.exp(-t * 7);
    lp += 0.25 * (v - lp); // felt-piano softness
    v = lp * env * vel;
    L[s0 + i] += v * (1 - pan) * 0.5;
    R[s0 + i] += v * (1 + pan) * 0.5;
  }
}

function addKick(L, R, start) {
  const s0 = Math.floor(start * SR);
  let ph = 0;
  for (let i = 0; i < SR * 0.4 && s0 + i < L.length; i++) {
    const t = i / SR;
    ph += (2 * Math.PI * (45 + 90 * Math.exp(-t * 30))) / SR;
    const v = Math.sin(ph) * Math.exp(-t * 9) * 0.9;
    L[s0 + i] += v;
    R[s0 + i] += v;
  }
}

function addSnare(L, R, start) {
  const s0 = Math.floor(start * SR);
  let lp = 0;
  for (let i = 0; i < SR * 0.22 && s0 + i < L.length; i++) {
    const t = i / SR;
    const n = rnd() * 2 - 1;
    lp += 0.4 * (n - lp);
    const v = (lp * 0.8 + 0.25 * Math.sin(2 * Math.PI * 190 * t)) * Math.exp(-t * 18) * 0.35;
    L[s0 + i] += v;
    R[s0 + i] += v;
  }
}

function addHat(L, R, start, vel) {
  const s0 = Math.floor(start * SR);
  let prev = 0;
  for (let i = 0; i < SR * 0.05 && s0 + i < L.length; i++) {
    const t = i / SR;
    const n = rnd() * 2 - 1;
    const hp = n - prev;
    prev = n;
    const v = hp * Math.exp(-t * 80) * 0.08 * vel;
    L[s0 + i] += v * 0.7;
    R[s0 + i] += v;
  }
}

function addBass(L, R, start, m, dur) {
  const f = midi(m);
  const s0 = Math.floor(start * SR);
  for (let i = 0; i < dur * SR && s0 + i < L.length; i++) {
    const t = i / SR;
    const env = Math.min(1, t * 200) * Math.exp(-t * 1.8);
    const v = (Math.sin(2 * Math.PI * f * t) + 0.2 * Math.sin(4 * Math.PI * f * t)) * env * 0.35;
    L[s0 + i] += v;
    R[s0 + i] += v;
  }
}

function music() {
  const L = buf(LEN);
  const R = buf(LEN);
  const bars = Math.ceil(LEN / BAR);
  let mel = 0;
  for (let b = 0; b < bars; b++) {
    const c = CHORDS[b % 4];
    const t0 = b * BAR;
    // chord stabs: beat 1 and the "and" of 2
    c.notes.forEach((m, k) => addPianoNote(L, R, t0 + k * 0.012, m, 0.32, 2.2, (k - 1.5) * 0.25));
    c.notes.forEach((m, k) => addPianoNote(L, R, t0 + BEAT * 1.5 + k * 0.01, m, 0.16, 1.4, (k - 1.5) * 0.25));
    addPianoNote(L, R, t0, c.bass + 12, 0.35, 2.4, 0);
    // sparse melody from bar 3
    if (b >= 2) {
      for (const pos of [2.5, 3.5]) {
        addPianoNote(L, R, t0 + BEAT * pos, MELODY[mel++ % MELODY.length], 0.22, 1.2, 0.3);
      }
    }
    // drums + bass from scene 5 on
    for (let beat = 0; beat < 4; beat++) {
      const t = t0 + beat * BEAT;
      if (t < DRUMS_IN - 0.01) continue;
      if (beat === 0 || beat === 2) addKick(L, R, t);
      if (beat === 1 || beat === 3) addSnare(L, R, t + 0.012); // lazy, behind the beat
      addHat(L, R, t, 1);
      addHat(L, R, t + BEAT / 2 + 0.03, 0.6);
      addBass(L, R, t, c.bass, BEAT * 0.95);
    }
  }
  // vinyl crackle
  for (let i = 0; i < L.length; i++) {
    if (rnd() < 0.00025) {
      const a = (rnd() - 0.5) * 0.25;
      L[i] += a;
      R[i] += a * 0.8;
    }
  }
  normalize([L, R], -3);
  return [L, R];
}

writeWav('public/sfx/pop.wav', pop());
writeWav('public/sfx/tap.wav', tap());
writeWav('public/sfx/ding.wav', ding());
writeWav('public/sfx/whoosh.wav', whoosh());
const [rl, rr] = roomtone();
writeWav('public/sfx/roomtone.wav', rl, rr);
const [ml, mr] = music();
writeWav('public/music/bed.wav', ml, mr);
