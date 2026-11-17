import React from 'react';
import {AbsoluteFill, Easing, interpolate, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {KineticText} from '../components/KineticText';
import {C, SAFE, SOFT} from '../theme';
import {useMouth} from '../voice';
import {ThoughtCloud, YanivShot, usePop} from './shared';

// SCENE 2 – PAIN – frames 84–165 (2.8–5.5 s), local 0–81
// Shoulders drop. Eye contact. Thought bubble: phone with a red X, then the
// brain drains to blank exactly on "ופתאום המוח ריק".

const BLANK_AT = 58; // local frame of "המוח ריק"

const PhoneX: React.FC<{p: number}> = ({p}) => (
  <svg viewBox="0 0 200 260" width={170} height={220} style={{transform: `scale(${p}) rotate(${(1 - p) * -30}deg)`}}>
    <rect x={40} y={20} width={120} height={220} rx={22} fill="#1A1B21" stroke={C.ink} strokeWidth={8} />
    <rect x={52} y={34} width={96} height={190} rx={14} fill={C.gray} />
    <circle cx={100} cy={128} r={22} fill={C.red} opacity={0.9} />
    <g stroke={C.red} strokeWidth={18} strokeLinecap="round">
      <path d="M 10 30 L 190 230" />
      <path d="M 190 30 L 10 230" />
    </g>
  </svg>
);

const Brain: React.FC<{p: number; blank: number; frame: number}> = ({p, blank, frame}) => {
  const fill = blank > 0.5 ? C.white : '#FF9FB0';
  return (
    <svg viewBox="0 0 240 200" width={230} height={190} style={{transform: `scale(${p})`}}>
      <path
        d="M 60 160 C 20 160 10 120 30 100 C 10 70 40 40 70 50 C 80 20 130 15 145 40 C 170 20 215 40 205 75 C 235 95 225 145 190 150 C 185 175 140 180 125 160 C 110 180 70 180 60 160 Z"
        fill={fill}
        stroke={C.ink}
        strokeWidth={8}
        strokeDasharray={blank > 0.5 ? '18 14' : undefined}
      />
      {blank < 0.5 ? (
        <g stroke="#D96A82" strokeWidth={6} fill="none" strokeLinecap="round">
          <path d="M 70 80 Q 95 70 100 95 Q 105 120 130 110" />
          <path d="M 140 60 Q 160 80 150 100" />
          <path d="M 80 130 Q 110 140 120 125" />
          <path d="M 165 120 Q 185 115 185 95" />
        </g>
      ) : (
        <g fill={C.ink}>
          {[0, 1, 2].map((i) => (
            <circle key={i} cx={90 + i * 30} cy={105} r={9} opacity={Math.floor(frame / 5) % 3 === i ? 1 : 0.25} />
          ))}
        </g>
      )}
    </svg>
  );
};

export const Scene2Pain: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const mouth = useMouth();

  const slump = spring({frame: frame - 2, fps, config: {damping: 8, stiffness: 120, mass: 0.8}});
  const zoom = interpolate(frame, [0, 81], [1.0, 1.1], {easing: Easing.out(Easing.quad)});
  const blank = spring({frame: frame - BLANK_AT, fps, config: SOFT});
  const expr = frame < BLANK_AT ? 'sad' : 'frozen';

  const cloud = usePop(8);
  const phoneP = usePop(12);
  const phoneOut = spring({frame: frame - 36, fps, config: SOFT});
  const brainP = usePop(40);

  return (
    <AbsoluteFill>
      <YanivShot zoom={zoom} x={-150} rot={-0.5} expression={expr} mouth={mouth} slump={slump} look={[0, 0]} y={110} />
      <ThoughtCloud cx={790} cy={590} w={400} h={310} scale={cloud}>
        <div style={{position: 'absolute', inset: 0, display: 'flex', alignItems: 'center', justifyContent: 'center'}}>
          {frame < 40 ? <PhoneX p={phoneP * (1 - phoneOut)} /> : <Brain p={brainP} blank={blank} frame={frame} />}
        </div>
      </ThoughtCloud>
      <KineticText
        text="כל בעל עסק מכיר את זה."
        cues={[6, 12, 18, 26, 34, 40]}
        appearAt={2}
        top={SAFE.top + 4}
        fontSize={82}
        tilt={1.5}
        holdLast={24}
        circle={{word: 3, at: 30, color: C.yellow}}
      />
    </AbsoluteFill>
  );
};
