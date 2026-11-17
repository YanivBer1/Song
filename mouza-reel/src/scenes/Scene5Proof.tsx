import React from 'react';
import {AbsoluteFill, random, useCurrentFrame} from 'remotion';
import {boilKey, wobblyPath} from '../components/Doodles';
import {SplitBeforeAfter} from '../components/SplitBeforeAfter';
import {Timer} from '../components/Timer';
import {C} from '../theme';
import {AFTER_WORDS, FinishedReel, RawTake, usePop} from './shared';

// SCENE 5 – BEFORE / AFTER – frames 390–540 (13.0–18.0 s), local 0–150
// Left: raw take (gray, frozen, silent). Right: same take after Mouza
// (captions word-by-word, B-roll, music). Timer 25:00 → 00:00 in 2 s.

// One continuous timeline for the finished edit, shared with scene 6.
export const REEL_CUES = AFTER_WORDS.map((_, i) => 8 + i * 17);
export const REEL_BROLL: {from: number; to: number; kind: 'hands' | 'shop'}[] = [
  {from: 40, to: 64, kind: 'hands'},
  {from: 92, to: 116, kind: 'shop'},
  {from: 200, to: 226, kind: 'hands'},
  {from: 246, to: 270, kind: 'shop'},
];
export const reelMouth = (t: number) => {
  // the edited take talks continuously, with natural breath gaps
  const gap = t % 40 > 34;
  return gap ? 0 : 0.15 + 0.75 * random(`reel${Math.floor(t / 2)}`);
};

const EqualsDoodle: React.FC = () => {
  const frame = useCurrentFrame();
  const s = usePop(58);
  if (frame < 58) return null;
  const k = boilKey(frame);
  const line = (y: number): [number, number][] => [
    [480, y],
    [520, y + 3],
    [560, y - 2],
    [600, y + 1],
  ];
  return (
    <svg viewBox="0 0 1080 1920" width={1080} height={1920} style={{position: 'absolute', inset: 0, overflow: 'visible'}}>
      <g transform={`translate(540 760) scale(${s}) translate(-540 -760)`}>
        <circle cx={540} cy={760} r={86} fill={C.bg} stroke={C.white} strokeWidth={8} />
        {[738, 782].map((y, i) => (
          <path key={y} d={wobblyPath(line(y), 6, `eq${i}${k}`)} stroke={C.yellow} strokeWidth={18} strokeLinecap="round" fill="none" />
        ))}
      </g>
    </svg>
  );
};

export const Scene5Proof: React.FC = () => {
  const frame = useCurrentFrame();
  return (
    <AbsoluteFill>
      <SplitBeforeAfter
        beforeSide="left"
        before={<RawTake t={frame} />}
        after={
          <FinishedReel
            t={frame}
            cues={REEL_CUES}
            broll={REEL_BROLL}
            captionTop={1140}
            captionLeft={290}
            captionWidth={500}
            captionSize={56}
            mouth={reelMouth(frame)}
          />
        }
      />
      <EqualsDoodle />
      <Timer start={20} x={810} y={404} />
    </AbsoluteFill>
  );
};
