import React from 'react';
import {random, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {BOUNCE, C, HEEBO, POP, SAFE, SAFE_WIDTH} from '../theme';

// Word-by-word kinetic headline (Hebrew, RTL).
//   current word  → yellow #FFD23F, punches in with an overshoot spring
//   previous words→ white
//   future words  → gray #8A8F98 at 40% opacity (already on screen = reader
//                   sees where the sentence is going, keeps eyes on screen)
// Words are separate inline-blocks inside a dir="rtl" flex row, so the
// first word sits on the RIGHT and punctuation stays glued to its word.

export const wordColor = (i: number, current: number) =>
  i === current ? C.yellow : i < current ? C.white : C.gray;

export const currentWordIndex = (frame: number, cues: number[], holdLast: number) => {
  let cur = -1;
  for (let i = 0; i < cues.length; i++) if (frame >= cues[i]) cur = i;
  // after the last word has been "said" for a while everything turns white
  if (cur === cues.length - 1 && frame > cues[cues.length - 1] + holdLast) return cues.length;
  return cur;
};

export type KineticTextProps = {
  text: string;
  /** frame (relative) at which each word becomes the current word */
  cues: number[];
  /** frame the whole block pops on (future words appear gray). default = cues[0] - 4 */
  appearAt?: number;
  top?: number;
  fontSize?: number;
  weight?: 500 | 700 | 800;
  /** how long the final word stays yellow before turning white */
  holdLast?: number;
  lineHeight?: number;
  /** extra rotation of the whole block, degrees – lo-fi */
  tilt?: number;
  maxWidth?: number;
  left?: number;
  align?: 'center' | 'right';
  /** render a translucent marker block behind the text */
  highlightBox?: boolean;
  /** optional exit frame: block springs away */
  exitAt?: number;
  /** anchor the block's BOTTOM edge at this y instead of its top (keeps text above the IG UI) */
  bottom?: number;
  /** hand-drawn circle around one word */
  circle?: {word: number; at: number; color?: string};
};

// Marker ellipse drawn INSIDE a word's box, so it follows real text layout.
const WordCircle: React.FC<{at: number; color: string; seed: string}> = ({at, color, seed}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  if (frame < at) return null;
  const p = spring({frame: (frame - at) * 1.4, fps, config: {damping: 18, stiffness: 90, mass: 0.6}});
  const k = Math.floor(frame / 4) % 6;
  const pts: string[] = [];
  for (let i = 0; i <= 26; i++) {
    const a = -Math.PI * 0.7 + (i / 26) * Math.PI * 2 * 1.16;
    const r = 1 + (i / 26) * 0.07 + (random(`${seed}${k}${i}`) - 0.5) * 0.05;
    pts.push(`${50 + Math.cos(a) * 50 * r},${50 + Math.sin(a) * 50 * r}`);
  }
  return (
    <svg
      viewBox="0 0 100 100"
      preserveAspectRatio="none"
      style={{position: 'absolute', left: '-14%', top: '-8%', width: '128%', height: '116%', overflow: 'visible', pointerEvents: 'none'}}
    >
      <polyline
        points={pts.join(' ')}
        fill="none"
        stroke={color}
        strokeWidth={10}
        strokeLinecap="round"
        strokeLinejoin="round"
        vectorEffect="non-scaling-stroke"
        pathLength={1}
        strokeDasharray={1}
        strokeDashoffset={1 - p}
      />
    </svg>
  );
};

export const KineticText: React.FC<KineticTextProps> = ({
  text,
  cues,
  appearAt,
  top,
  fontSize = 88,
  weight = 700,
  holdLast = 18,
  lineHeight = 1.12,
  tilt = 0,
  maxWidth = SAFE_WIDTH,
  left = SAFE.left,
  align = 'center',
  highlightBox = false,
  exitAt,
  bottom,
  circle,
}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const words = text.split(' ');
  const appear = appearAt ?? cues[0] - 4;
  if (frame < appear) return null;

  const blockIn = spring({frame: frame - appear, fps, config: BOUNCE});
  const blockOut = exitAt !== undefined ? spring({frame: frame - exitAt, fps, config: POP}) : 0;
  if (blockOut > 0.98) return null;
  const cur = currentWordIndex(frame, cues, holdLast);

  return (
    <div
      dir="rtl"
      style={{
        position: 'absolute',
        ...(bottom !== undefined ? {bottom: 1920 - bottom} : {top}),
        left,
        width: maxWidth,
        display: 'flex',
        flexWrap: 'wrap',
        justifyContent: align === 'center' ? 'center' : 'flex-start',
        alignItems: 'baseline',
        columnGap: fontSize * 0.26,
        rowGap: fontSize * 0.04,
        direction: 'rtl',
        textAlign: align,
        fontFamily: HEEBO,
        fontWeight: weight,
        fontSize,
        lineHeight,
        transform: `rotate(${tilt}deg) scale(${(0.6 + 0.4 * blockIn) * (1 - blockOut)}) translateY(${blockOut * -80}px)`,
        transformOrigin: '50% 50%',
        opacity: Math.min(1, blockIn * 1.5) * (1 - blockOut),
      }}
    >
      {words.map((w, i) => {
        const isCur = i === cur;
        const s = spring({frame: frame - (cues[i] ?? 9999), fps, config: POP});
        const punch = isCur ? 1 + 0.32 * (1 - s) + 0.06 : 1;
        const wordTilt = (random(`kt${text}${i}`) - 0.5) * 6;
        return (
          <span
            key={i}
            dir="rtl"
            style={{
              display: 'inline-block',
              position: 'relative',
              color: wordColor(i, cur),
              opacity: i > cur ? 0.4 : 1,
              transform: `scale(${punch}) rotate(${isCur ? wordTilt : wordTilt * 0.3}deg) translateY(${isCur ? (1 - s) * -18 : 0}px)`,
              WebkitTextStroke: `${Math.round(fontSize * 0.12)}px ${C.bg}`,
              paintOrder: 'stroke fill',
              textShadow: `0 ${fontSize * 0.07}px 0 rgba(0,0,0,0.55)`,
              whiteSpace: 'nowrap',
              unicodeBidi: 'isolate',
            }}
          >
            {highlightBox && isCur ? (
              <span
                style={{
                  position: 'absolute',
                  inset: `${fontSize * 0.12}px -${fontSize * 0.12}px ${fontSize * 0.02}px -${fontSize * 0.12}px`,
                  background: C.teal,
                  opacity: 0.28,
                  borderRadius: 14,
                  transform: `rotate(${-wordTilt}deg)`,
                  zIndex: -1,
                }}
              />
            ) : null}
            {w}
            {circle && circle.word === i ? <WordCircle at={circle.at} color={circle.color ?? C.yellow} seed={`${text}${i}`} /> : null}
          </span>
        );
      })}
    </div>
  );
};
