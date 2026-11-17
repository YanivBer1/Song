import React from 'react';
import {AbsoluteFill, interpolate, spring, useCurrentFrame, useVideoConfig, Easing} from 'remotion';
import type {Expression} from '../components/CharacterYaniv';
import {KineticText} from '../components/KineticText';
import {MouzaWordmark} from '../components/MouzaLogo';
import {LOOP_FRAMING} from '../hooks';
import {LOOP_LAUGH_AT} from '../timing';
import {C, POP, SAFE} from '../theme';
import {useMouth} from '../voice';
import {PaperPlane, YanivShot, usePop} from './shared';

// SCENE 7 – LOOP + SEND – frames 690–810 (23.0–27.0 s), local 0–120
// Same close-up as scene 1. Teasing point at camera, "you know who",
// laugh – and we cut mid-laugh on EXACTLY the frame-0 framing so the
// Reel loops seamlessly back into "ניסיון 1".

export const Scene7Loop: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const mouth = useMouth();

  const laughing = frame >= LOOP_LAUGH_AT;
  let expr: Expression = 'smirk';
  if (frame >= 62 && frame < LOOP_LAUGH_AT) expr = 'smile';
  if (laughing) expr = 'laugh';

  // framing: starts wide-ish, jump-cuts in on "אתם יודעים מי", lands on the
  // loop framing for the laugh (identical to frame 0 of the hook).
  const framing = frame < 62 ? {zoom: 1.0, x: -70, rot: 0.8} : frame < LOOP_LAUGH_AT ? {zoom: 1.22, x: -40, rot: -1} : LOOP_FRAMING;
  const cutAt = frame < 62 ? 0 : frame < LOOP_LAUGH_AT ? 62 : LOOP_LAUGH_AT;
  const cs = spring({frame: frame - cutAt, fps, config: POP});
  const zoom = framing.zoom * (laughing ? 1 : 1 + 0.04 * (1 - cs));

  const point = frame < LOOP_LAUGH_AT ? spring({frame: frame - 4, fps, config: POP}) : 0;
  const laughBounce = laughing ? 1 + 0.035 * Math.sin((frame - LOOP_LAUGH_AT) * 1.3) : 0.95 + 0.05 * cs;

  // paper plane = "send"
  const planeT = interpolate(frame, [40, 70], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp', easing: Easing.in(Easing.quad)});
  const logo = usePop(70);

  return (
    <AbsoluteFill>
      <YanivShot
        zoom={zoom}
        x={framing.x}
        rot={framing.rot}
        expression={expr}
        mouth={laughing ? 0 : mouth}
        point={point}
        squash={laughBounce}
        shake={3.5}
      />
      {frame >= 40 && frame < 72 ? (
        <PaperPlane x={330 + planeT * 760} y={1040 - planeT * 800 + Math.sin(planeT * 9) * 40} rot={-20 - planeT * 15} s={1.1} />
      ) : null}
      <KineticText
        text="שלחו את זה למי שתמיד אומר 'מחר אני מצלם.'"
        cues={[2, 8, 12, 18, 24, 31, 38, 45, 52]}
        appearAt={0}
        top={SAFE.top + 4}
        fontSize={86}
        tilt={-1.5}
        holdLast={30}
        circle={{word: 6, at: 40, color: C.red}}
      />
      {frame >= 70 ? (
        <div style={{position: 'absolute', left: 540, top: SAFE.bottom - 84, transform: `translateX(-50%) scale(${logo * 0.9})`}}>
          <MouzaWordmark height={58} />
        </div>
      ) : null}
    </AbsoluteFill>
  );
};
