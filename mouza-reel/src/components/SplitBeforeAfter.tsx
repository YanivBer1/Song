import React from 'react';
import {AbsoluteFill, random, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {BOUNCE, C, HEEBO, POP, SAFE} from '../theme';
import {boilKey, wobblyPath} from './Doodles';

// Split screen. Default per brief: LEFT = before (flat, gray, boring),
// RIGHT = after (color, captions, music). Flip with beforeSide="right".
// Each side receives a full 1080×1920 scene and shows its center crop,
// so "same person, same phone, same room" is literally the same framing.

const Label: React.FC<{text: string; bg: string; fg: string; x: number; at: number; tilt: number}> = ({text, bg, fg, x, at, tilt}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const s = spring({frame: frame - at, fps, config: POP});
  if (frame < at) return null;
  return (
    <div
      dir="rtl"
      style={{
        position: 'absolute',
        top: SAFE.top + 8,
        left: x,
        transform: `translateX(-50%) rotate(${tilt}deg) scale(${s})`,
        background: bg,
        color: fg,
        fontFamily: HEEBO,
        fontWeight: 800,
        fontSize: 84,
        lineHeight: 1,
        padding: '10px 34px 18px',
        borderRadius: 20,
        border: `5px solid ${C.bg}`,
        boxShadow: `7px 9px 0 ${C.bg}`,
        whiteSpace: 'nowrap',
      }}
    >
      {text}
    </div>
  );
};

export const SplitBeforeAfter: React.FC<{
  before: React.ReactNode;
  after: React.ReactNode;
  beforeSide?: 'left' | 'right';
  beforeLabel?: string;
  afterLabel?: string;
}> = ({before, after, beforeSide = 'left', beforeLabel = 'לפני', afterLabel = 'אחרי'}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const inL = spring({frame: frame + 1, fps, config: BOUNCE});
  const inR = spring({frame: frame - 3, fps, config: BOUNCE});
  const k = boilKey(frame);
  const divider: [number, number][] = Array.from({length: 12}, (_, i) => [540 + (random(`dv${i}`) - 0.5) * 10, i * 175]);

  const leftNode = beforeSide === 'left' ? before : after;
  const rightNode = beforeSide === 'left' ? after : before;

  const half = (node: React.ReactNode, side: 'left' | 'right', p: number) => (
    <div
      style={{
        position: 'absolute',
        top: 0,
        left: side === 'left' ? 0 : 540,
        width: 540,
        height: 1920,
        overflow: 'hidden',
        transform: `translateX(${(1 - p) * (side === 'left' ? -560 : 560)}px)`,
      }}
    >
      <div style={{position: 'absolute', top: 0, left: -270, width: 1080, height: 1920}}>{node}</div>
    </div>
  );

  return (
    <AbsoluteFill style={{background: C.bg}}>
      {half(leftNode, 'left', inL)}
      {half(rightNode, 'right', inR)}
      <svg viewBox="0 0 1080 1920" width={1080} height={1920} style={{position: 'absolute', inset: 0}}>
        <path d={wobblyPath(divider, 8, `div${k}`)} stroke={C.white} strokeWidth={12} fill="none" strokeLinecap="round" />
        <path d={wobblyPath(divider, 8, `div2${k}`)} stroke={C.bg} strokeWidth={4} fill="none" strokeLinecap="round" opacity={0.6} />
      </svg>
      <Label
        text={beforeLabel}
        bg={C.gray}
        fg={C.bg}
        x={beforeSide === 'left' ? 270 : 810}
        at={4}
        tilt={-5}
      />
      <Label text={afterLabel} bg={C.teal} fg={C.bg} x={beforeSide === 'left' ? 810 : 270} at={10} tilt={4} />
    </AbsoluteFill>
  );
};
