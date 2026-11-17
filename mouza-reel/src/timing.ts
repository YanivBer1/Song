// ============================================================================
//  VO TIMING MAP – the single source of truth for sync.
//  All numbers are FRAMES at 30 fps. Scene frames are absolute; everything
//  inside a scene (offset, talk windows, word cues) is RELATIVE to scene start.
//
//  After you record your real phone VO, drop the files in public/vo/ and nudge
//  the numbers here while scrubbing in Remotion Studio. Nothing else needs
//  to change: mouths, captions and highlights all read from this file.
// ============================================================================

export type SceneKey = 'hook' | 'pain' | 'solution' | 'retake' | 'proof' | 'payoff' | 'loop';

export const SCENES: Record<SceneKey, {from: number; dur: number; label: string}> = {
  hook: {from: 0, dur: 84, label: '1 HOOK 0.0–2.8s'},
  pain: {from: 84, dur: 81, label: '2 PAIN 2.8–5.5s'},
  solution: {from: 165, dur: 105, label: '3 SOLUTION 5.5–9.0s'},
  retake: {from: 270, dur: 120, label: '4 RETAKE 9.0–13.0s'},
  proof: {from: 390, dur: 150, label: '5 BEFORE/AFTER 13.0–18.0s'},
  payoff: {from: 540, dur: 150, label: '6 PAYOFF 18.0–23.0s'},
  loop: {from: 690, dur: 120, label: '7 LOOP+SEND 23.0–27.0s'},
};

export type VoLine = {
  /** file name inside public/vo/ */
  file: string;
  /** frames after scene start where the recording begins */
  offset: number;
  /** exact line to record */
  text: string;
  /** frames (relative to scene) where the mouth moves when no recording exists yet */
  talk: [number, number][];
};

// Scene 1 VO lives in src/hooks.ts because it changes per A/B variant.
export const VO: Record<Exclude<SceneKey, 'hook'>, VoLine> = {
  pain: {
    file: 'scene2-pain.wav',
    offset: 0,
    text: 'אתם מכירים את זה, נכון? יושבים לצלם, ופתאום המוח ריק.',
    talk: [
      [1, 34],
      [41, 78],
    ],
  },
  solution: {
    file: 'scene3-solution.wav',
    offset: 4,
    text: 'אז עשיתי משהו אחר. כתבתי משפט אחד במוזה.',
    talk: [
      [4, 38],
      [46, 92],
    ],
  },
  retake: {
    file: 'scene4-retake.wav',
    offset: 2,
    text: 'הנה, פה טעיתי. לא מצלמים הכול מחדש. רק את הקטע הזה.',
    talk: [
      [2, 30],
      [38, 72],
      [80, 112],
    ],
  },
  proof: {
    file: 'scene5-proof.wav',
    offset: 8,
    text: 'וזה מה שיצא. אותו בן אדם, אותו טלפון.',
    talk: [
      [8, 40],
      [56, 112],
    ],
  },
  payoff: {
    file: 'scene6-payoff.wav',
    offset: 54,
    text: 'אתם מצלמים. מוזה עורכת.',
    talk: [
      [54, 80],
      [92, 122],
    ],
  },
  loop: {
    file: 'scene7-loop.wav',
    offset: 0,
    text: 'שלחו את זה למי שתמיד אומר מחר אני מצלם. אתם יודעים מי.',
    talk: [
      [1, 62],
      [68, 88],
    ],
  },
};

// Laugh at the very end of scene 7 (relative frames). Music cuts here.
export const LOOP_LAUGH_AT = 88;

// Music volume automation (absolute frames → linear gain).
// No music in the hook. Low piano at 2.8s, lift at 9s, full at 13s,
// finished-video music at 18s then duck under "אתם מצלמים", cut on the laugh.
export const MUSIC_KEYS: [number, number][] = [
  [0, 0],
  [84, 0],
  [96, 0.16],
  [270, 0.16],
  [282, 0.24],
  [390, 0.24],
  [410, 0.62],
  [540, 0.62],
  [546, 0.72],
  [590, 0.72],
  [600, 0.2],
  [690, 0.2],
  [SCENES.loop.from + LOOP_LAUGH_AT, 0.2],
  [SCENES.loop.from + LOOP_LAUGH_AT + 1, 0],
  [810, 0],
];

/** helper: evenly spaced word cues */
export const cues = (count: number, start: number, step: number) =>
  Array.from({length: count}, (_, i) => start + i * step);
