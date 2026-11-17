import React from 'react';
import {AbsoluteFill, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {Counter} from '../components/Counter';
import {KineticText} from '../components/KineticText';
import type {HookConfig} from '../hooks';
import {C, POP, SAFE} from '../theme';
import {useMouth} from '../voice';
import {Flash, YanivShot} from './shared';

// SCENE 1 – HOOK – frames 0–84 (0.0–2.8 s)
// Hard open on Yaniv mid-fail. No intro, no logo. Jump cut every 9–12
// frames (0.3–0.4 s), counter slams, then the confession.

const Viewfinder: React.FC = () => {
  const frame = useCurrentFrame();
  const L = 70;
  const m = 40;
  const corner = (x: number, y: number, sx: number, sy: number) => `M ${x} ${y + sy * L} L ${x} ${y} L ${x + sx * L} ${y}`;
  return (
    <svg viewBox="0 0 1080 1920" width={1080} height={1920} style={{position: 'absolute', inset: 0}}>
      <g stroke={C.white} strokeWidth={7} fill="none" strokeLinecap="round" opacity={0.75}>
        <path d={corner(m, 240, 1, 1)} />
        <path d={corner(1080 - m, 240, -1, 1)} />
        <path d={corner(m, 1380, 1, -1)} />
        <path d={corner(1080 - m, 1380, -1, -1)} />
      </g>
      <circle cx={m + 40} cy={300} r={16} fill={C.red} opacity={frame % 20 < 12 ? 1 : 0.25} />
    </svg>
  );
};

export const Scene1Hook: React.FC<{hook: HookConfig}> = ({hook}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const mouth = useMouth();

  let ci = 0;
  hook.cuts.forEach((c, i) => {
    if (frame >= c.at) ci = i;
  });
  const cut = hook.cuts[ci];
  // each jump cut lands with a tiny punch + squash (spring overshoot)
  const s = ci === 0 ? 1 : spring({frame: frame - cut.at, fps, config: POP});
  const zoom = cut.zoom * (1 + 0.045 * (1 - s));
  const squash = 0.94 + 0.06 * s;
  const pointS = cut.point ? spring({frame: frame - cut.at + (ci === 0 ? 10 : 0), fps, config: POP}) : 0;
  const phoneS = cut.phone ? spring({frame: frame - cut.at, fps, config: POP}) : 0;

  return (
    <AbsoluteFill>
      <YanivShot
        zoom={zoom}
        x={cut.x}
        rot={cut.rot}
        expression={cut.expr}
        mouth={mouth}
        squash={squash}
        point={pointS}
        phone={phoneS}
        shake={3.5}
      />
      <Viewfinder />
      <Counter items={hook.counter.items} icon={hook.counter.icon} x={750} y={SAFE.bottom - 96} />
      <KineticText
        text={hook.text}
        cues={hook.cues}
        appearAt={-6}
        top={SAFE.top + 4}
        fontSize={hook.fontSize}
        tilt={-2}
        holdLast={30}
        circle={{word: hook.circleWord, at: hook.cues[hook.circleWord] + 3, color: C.yellow}}
      />
      <Flash at={hook.counter.items.map((i) => i.at).filter((a) => a > 0)} />
    </AbsoluteFill>
  );
};
