import React from 'react';
import {spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {HEEBO, POP} from '../theme';
import {currentWordIndex, wordColor} from './KineticText';

// Auto-caption style used INSIDE the "finished Mouza video" (scenes 5 & 6):
// short pages of 2–3 words, chunky Heebo with a dark outline, the same
// yellow/white/gray-40% word-state rule as the headline text.

export type CaptionProps = {
  words: string[];
  cues: number[];
  /** words per caption page */
  chunk?: number;
  top: number;
  left: number;
  width: number;
  fontSize?: number;
};

export const CaptionWordByWord: React.FC<CaptionProps> = ({words, cues, chunk = 3, top, left, width, fontSize = 64}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  if (frame < cues[0]) return null;
  const cur = Math.min(currentWordIndex(frame, cues, 9999), words.length - 1);
  const page = Math.floor(cur / chunk);
  const pageStart = page * chunk;
  const pageWords = words.slice(pageStart, pageStart + chunk);
  const pageIn = spring({frame: frame - cues[pageStart], fps, config: POP});

  return (
    <div
      dir="rtl"
      style={{
        position: 'absolute',
        top,
        left,
        width,
        display: 'flex',
        flexWrap: 'wrap',
        justifyContent: 'center',
        columnGap: fontSize * 0.25,
        direction: 'rtl',
        textAlign: 'center',
        fontFamily: HEEBO,
        fontWeight: 800,
        fontSize,
        lineHeight: 1.15,
        transform: `scale(${0.7 + 0.3 * pageIn})`,
      }}
    >
      {pageWords.map((w, k) => {
        const i = pageStart + k;
        const isCur = i === cur;
        const s = spring({frame: frame - cues[i], fps, config: POP});
        return (
          <span
            key={i}
            dir="rtl"
            style={{
              display: 'inline-block',
              color: wordColor(i, cur),
              opacity: i > cur ? 0.4 : 1,
              transform: `scale(${isCur ? 1.12 + 0.2 * (1 - s) : 1})`,
              WebkitTextStroke: `${Math.round(fontSize * 0.14)}px #000`,
              paintOrder: 'stroke fill',
              whiteSpace: 'nowrap',
              unicodeBidi: 'isolate',
            }}
          >
            {w}
          </span>
        );
      })}
    </div>
  );
};
