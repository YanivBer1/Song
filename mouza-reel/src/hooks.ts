import type {Expression} from './components/CharacterYaniv';
import type {CounterIcon, CounterItem} from './components/Counter';
import type {VoLine} from './timing';

// ============================================================================
//  SCENE 1 HOOK VARIANTS (0–84 frames). Everything from frame 84 on is the
//  shared body and is IDENTICAL across all compositions.
//  Per variant only: on-screen text, VO line, counter animation, cut list.
// ============================================================================

export type HookCut = {
  at: number; // frame (0–83) – a hard "jump cut" every 0.2–0.4 s
  zoom: number; // punch-in
  x: number; // horizontal reframe (px)
  rot: number; // dutch angle (deg)
  expr: Expression;
  point?: boolean;
  phone?: boolean;
};

export type HookConfig = {
  id: string;
  name: string;
  text: string; // on-screen text, exact Hebrew
  cues: number[]; // frame each word becomes "current"
  fontSize: number;
  circleWord: number; // index of the word that gets the doodle circle
  vo: VoLine;
  counter: {icon: CounterIcon; items: CounterItem[]};
  cuts: HookCut[];
};

// The framing of frame 0 – scene 7 returns to EXACTLY this for a seamless loop.
export const LOOP_FRAMING = {zoom: 1.12, x: -46, rot: -1.5};

const CONFESSION_CUTS: HookCut[] = [
  {at: 0, ...LOOP_FRAMING, expr: 'frozen'},
  {at: 9, zoom: 1.0, x: -70, rot: 1.2, expr: 'wince'},
  {at: 20, zoom: 1.24, x: -30, rot: -2.2, expr: 'sad'},
  {at: 30, zoom: 1.04, x: -80, rot: 0.6, expr: 'embarrassed'},
  {at: 40, zoom: 1.3, x: -40, rot: 2, expr: 'laugh'},
  {at: 51, zoom: 1.06, x: -64, rot: -0.8, expr: 'embarrassed'},
  {at: 63, zoom: 1.18, x: -50, rot: 1, expr: 'smirk', point: true},
  {at: 74, zoom: 1.02, x: -62, rot: 0, expr: 'embarrassed', point: true},
];

const ATTEMPTS: CounterItem[] = [
  {label: 'ניסיון 1', at: 0},
  {label: 'ניסיון 7', at: 20},
  {label: 'ניסיון 11', at: 40},
];

export const HOOKS: Record<string, HookConfig> = {
  // ---- MAIN: Mouza11Attempts ----
  main: {
    id: 'main',
    name: 'Mouza11Attempts',
    text: 'אני לא אמיתי. אני מצויר.',
    cues: [0, 8, 15, 28, 36],
    fontSize: 104,
    circleWord: 4,
    vo: {
      file: 'hook-A.wav',
      offset: 0,
      text: 'כן, אני מצויר. כי האמיתי פחד לצלם. כמוך.',
      talk: [
        [1, 22],
        [27, 60],
        [64, 80],
      ],
    },
    counter: {icon: 'rec', items: ATTEMPTS},
    cuts: CONFESSION_CUTS,
  },

  // ---- A: Confession ----
  A: {
    id: 'A',
    name: 'MouzaHookA',
    text: 'כן, אני מצויר. כי האמיתי פחד לצלם. כמוך.',
    cues: [0, 7, 13, 27, 32, 42, 50, 66],
    fontSize: 84,
    circleWord: 7,
    vo: {
      file: 'hook-A.wav',
      offset: 0,
      text: 'כן, אני מצויר. כי האמיתי פחד לצלם. כמוך.',
      talk: [
        [1, 22],
        [27, 60],
        [64, 80],
      ],
    },
    counter: {icon: 'rec', items: ATTEMPTS},
    cuts: CONFESSION_CUTS,
  },

  // ---- B: Direct call-out ----
  B: {
    id: 'B',
    name: 'MouzaHookB',
    text: 'בעל עסק? תעצור רגע. זה יחסוך לך 19 יום של התכתבויות עם עורך.',
    cues: [0, 5, 12, 17, 26, 30, 36, 40, 46, 51, 55, 66, 71],
    fontSize: 70,
    circleWord: 7,
    vo: {
      file: 'hook-B.wav',
      offset: 0,
      text: 'בעל עסק? תעצור רגע. זה יחסוך לך 19 יום של התכתבויות עם עורך.',
      talk: [
        [0, 22],
        [25, 82],
      ],
    },
    counter: {
      icon: 'chat',
      items: [
        {label: 'יום 1', at: 0},
        {label: 'יום 6', at: 14},
        {label: 'יום 13', at: 28},
        {label: 'יום 19', at: 42},
      ],
    },
    cuts: [
      {at: 0, ...LOOP_FRAMING, expr: 'surprised', point: true},
      {at: 8, zoom: 1.26, x: -40, rot: 2, expr: 'surprised', point: true},
      {at: 17, zoom: 1.04, x: -70, rot: -1, expr: 'smirk', point: true},
      {at: 26, zoom: 1.2, x: -50, rot: 1.5, expr: 'neutral'},
      {at: 36, zoom: 1.0, x: -64, rot: -1.5, expr: 'sad'},
      {at: 46, zoom: 1.3, x: -36, rot: 2.2, expr: 'wince'},
      {at: 58, zoom: 1.08, x: -60, rot: 0, expr: 'sad'},
      {at: 70, zoom: 1.16, x: -48, rot: -1, expr: 'smirk', point: true},
    ],
  },

  // ---- C: Shame ----
  C: {
    id: 'C',
    name: 'MouzaHookC',
    text: 'אני שונא להצטלם. תראו מה יצא לי בכל זאת.',
    cues: [0, 7, 14, 30, 36, 41, 46, 52, 58],
    fontSize: 88,
    circleWord: 2,
    vo: {
      file: 'hook-C.wav',
      offset: 0,
      text: 'אני שונא להצטלם. תראו מה יצא לי בכל זאת.',
      talk: [
        [1, 26],
        [30, 70],
      ],
    },
    counter: {
      icon: 'trash',
      items: [
        {label: 'מחיקה 1', at: 0},
        {label: 'מחיקה 6', at: 18},
        {label: 'מחיקה 14', at: 36},
      ],
    },
    cuts: [
      {at: 0, ...LOOP_FRAMING, expr: 'wince'},
      {at: 10, zoom: 1.28, x: -40, rot: 2, expr: 'wince'},
      {at: 20, zoom: 1.02, x: -74, rot: -1, expr: 'sad'},
      {at: 30, zoom: 1.2, x: -50, rot: 1.2, expr: 'embarrassed'},
      {at: 42, zoom: 1.06, x: -60, rot: -2, expr: 'smirk'},
      {at: 54, zoom: 1.24, x: -44, rot: 1.6, expr: 'smile', phone: true},
      {at: 66, zoom: 1.08, x: -62, rot: 0, expr: 'proud', phone: true},
    ],
  },

  // ---- D: Anti-ad ----
  D: {
    id: 'D',
    name: 'MouzaHookD',
    text: 'אל תשלחו את זה לעורך שלכם. באמת, אל תשלחו.',
    cues: [0, 7, 14, 18, 23, 31, 46, 58, 64],
    fontSize: 88,
    circleWord: 8,
    vo: {
      file: 'hook-D.wav',
      offset: 0,
      text: 'אל תשלחו את זה לעורך שלכם. באמת, אל תשלחו.',
      talk: [
        [1, 38],
        [44, 76],
      ],
    },
    counter: {
      icon: 'receipt',
      items: [
        {label: '800 ₪', at: 0},
        {label: '1,600 ₪', at: 20},
        {label: '2,400 ₪', at: 40},
      ],
    },
    cuts: [
      {at: 0, ...LOOP_FRAMING, expr: 'smirk', point: true},
      {at: 10, zoom: 1.24, x: -40, rot: -2, expr: 'smirk', point: true},
      {at: 21, zoom: 1.02, x: -70, rot: 1, expr: 'neutral', point: true},
      {at: 32, zoom: 1.18, x: -50, rot: 2, expr: 'surprised'},
      {at: 44, zoom: 1.3, x: -36, rot: -1.5, expr: 'smirk', point: true},
      {at: 56, zoom: 1.04, x: -64, rot: 0.5, expr: 'wince', point: true},
      {at: 68, zoom: 1.14, x: -50, rot: -1, expr: 'laugh'},
    ],
  },

  // ---- E: Question ----
  E: {
    id: 'E',
    name: 'MouzaHookE',
    text: "כמה פעמים כבר אמרת 'מחר אני מצלם' ולא צילמת?",
    cues: [0, 7, 15, 21, 30, 37, 43, 55, 62],
    fontSize: 84,
    circleWord: 4,
    vo: {
      file: 'hook-E.wav',
      offset: 0,
      text: "כמה פעמים כבר אמרת 'מחר אני מצלם' ולא צילמת?",
      talk: [
        [1, 28],
        [30, 52],
        [55, 76],
      ],
    },
    counter: {
      icon: 'calendar',
      items: [
        {label: 'פעם 1', at: 0},
        {label: 'פעם 12', at: 16},
        {label: 'פעם 31', at: 30},
        {label: 'פעם 47', at: 44},
      ],
    },
    cuts: [
      {at: 0, ...LOOP_FRAMING, expr: 'smirk'},
      {at: 10, zoom: 1.22, x: -44, rot: 1.6, expr: 'smirk'},
      {at: 21, zoom: 1.02, x: -72, rot: -1, expr: 'neutral'},
      {at: 30, zoom: 1.28, x: -40, rot: 2, expr: 'embarrassed'},
      {at: 43, zoom: 1.06, x: -64, rot: -1.6, expr: 'sad'},
      {at: 55, zoom: 1.2, x: -50, rot: 1, expr: 'smirk', point: true},
      {at: 68, zoom: 1.04, x: -60, rot: 0, expr: 'smirk', point: true},
    ],
  },
};
