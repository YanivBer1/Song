import {loadFont} from '@remotion/google-fonts/Heebo';
import {loadFont as loadLocalFont} from '@remotion/fonts';
import {staticFile} from 'remotion';

// Heebo Bold (700) for headlines, Heebo Medium (500) for body, 800 for counters.
// Default: @remotion/google-fonts. Offline mode (no internet / CI / proxies):
//   REMOTION_LOCAL_FONTS=1 npx remotion render ...   → uses public/fonts/*.ttf
const LOCAL = typeof process !== 'undefined' && process.env.REMOTION_LOCAL_FONTS === '1';

export const HEEBO: string = LOCAL
  ? (() => {
      for (const w of ['500', '700', '800']) {
        loadLocalFont({family: 'HeeboLocal', url: staticFile(`fonts/Heebo-${w}.ttf`), weight: w});
      }
      return 'HeeboLocal';
    })()
  : loadFont('normal', {weights: ['500', '700', '800'], subsets: ['hebrew', 'latin']}).fontFamily;

// Exact brand palette from the brief.
export const C = {
  bg: '#0E0F13',
  teal: '#00D1A9',
  yellow: '#FFD23F',
  white: '#FFFFFF',
  gray: '#8A8F98',
  red: '#FF4D4D',
  // non-text helpers derived from the palette (room, ink lines, skin)
  ink: '#0A0B0E',
  wall: '#262A33',
  wallDark: '#1C1F27',
  skin: '#E9B48F',
  skinShade: '#C98E6A',
  hair: '#2A221E',
  shirt: '#121318',
  shirtEdge: '#3A3F4B',
} as const;

export const W = 1080;
export const H = 1920;
export const FPS = 30;
export const DURATION = 810;

// Instagram safe zone: top 14%, bottom 35%, 6% each side are kept clear of text.
export const SAFE = {
  top: Math.round(H * 0.14), // 269
  bottom: Math.round(H * 0.65), // 1248  (everything below is IG UI)
  left: Math.round(W * 0.06), // 65
  right: Math.round(W * 0.94), // 1015
};
export const SAFE_WIDTH = SAFE.right - SAFE.left; // 950

// Spring presets. Low damping = overshoot = the bouncy "hyperframes" feel.
export const POP = {damping: 9, stiffness: 220, mass: 0.6};
export const BOUNCE = {damping: 7, stiffness: 160, mass: 0.7};
export const SOFT = {damping: 14, stiffness: 120, mass: 0.8};
