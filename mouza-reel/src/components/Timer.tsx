import React from 'react';
import {Easing, interpolate, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, HEEBO, POP} from '../theme';

// Stopwatch sticker: 25:00 → 00:00 in 2 seconds (60 frames), then stops,
// slams, and shows the "25 דקות." stamp.

export const Timer: React.FC<{start: number; x: number; y: number; duration?: number; stamp?: string}> = ({
  start,
  x,
  y,
  duration = 60,
  stamp = '25 דקות.',
}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  if (frame < start) return null;
  const inS = spring({frame: frame - start, fps, config: POP});
  const secs = Math.round(
    interpolate(frame, [start, start + duration], [25 * 60, 0], {
      extrapolateLeft: 'clamp',
      extrapolateRight: 'clamp',
      easing: Easing.inOut(Easing.cubic),
    }),
  );
  const mm = String(Math.floor(secs / 60)).padStart(2, '0');
  const ss = String(secs % 60).padStart(2, '0');
  const done = frame >= start + duration;
  const slam = spring({frame: frame - start - duration, fps, config: POP});
  const shake = !done ? Math.sin(frame * 2.3) * 3 : 0;

  return (
    <div style={{position: 'absolute', left: x, top: y, transform: `translate(-50%, 0) rotate(${-4 + shake}deg) scale(${inS * (done ? 1 + 0.18 * (1 - slam) : 1)})`}}>
      <div
        dir="rtl"
        style={{
          display: 'flex',
          alignItems: 'center',
          gap: 14,
          background: C.bg,
          border: `5px solid ${done ? C.teal : C.white}`,
          borderRadius: 26,
          padding: '8px 22px',
          boxShadow: `6px 8px 0 rgba(0,0,0,0.6)`,
        }}
      >
        <svg width={54} height={54} viewBox="0 0 54 54">
          <circle cx={27} cy={30} r={20} fill="none" stroke={done ? C.teal : C.white} strokeWidth={5} />
          <path d="M 21 6 H 33 M 27 6 V 10" stroke={done ? C.teal : C.white} strokeWidth={5} strokeLinecap="round" />
          <path
            d={`M 27 30 L ${27 + 13 * Math.sin((secs / 1500) * Math.PI * 2 * 6)} ${30 - 13 * Math.cos((secs / 1500) * Math.PI * 2 * 6)}`}
            stroke={C.yellow}
            strokeWidth={5}
            strokeLinecap="round"
          />
        </svg>
        <span
          dir="rtl"
          style={{
            fontFamily: HEEBO,
            fontWeight: 800,
            fontSize: 66,
            color: done ? C.teal : C.white,
            fontVariantNumeric: 'tabular-nums',
            lineHeight: 1,
            unicodeBidi: 'isolate',
          }}
        >
          {mm}:{ss}
        </span>
      </div>
      {done ? (
        <div
          dir="rtl"
          style={{
            marginTop: 10,
            textAlign: 'center',
            fontFamily: HEEBO,
            fontWeight: 800,
            fontSize: 58,
            color: C.bg,
            background: C.yellow,
            borderRadius: 14,
            padding: '2px 18px 6px',
            transform: `rotate(5deg) scale(${slam})`,
            border: `4px solid ${C.bg}`,
          }}
        >
          {stamp}
        </div>
      ) : null}
    </div>
  );
};
