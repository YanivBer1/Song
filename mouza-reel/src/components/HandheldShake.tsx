import React from 'react';
import {AbsoluteFill, useCurrentFrame} from 'remotion';

// Phone-in-hand camera wobble: 2–4 px drift + micro-rotation, plus an
// optional punch-in zoom. Deterministic (sum of sines), so every render
// is identical – no Math.random, no setTimeout.

export const shakeAt = (frame: number, amp = 3) => {
  const x = amp * (0.6 * Math.sin(frame * 0.31) + 0.3 * Math.sin(frame * 0.87 + 1.3) + 0.1 * Math.sin(frame * 2.1));
  const y = amp * (0.6 * Math.sin(frame * 0.27 + 2) + 0.3 * Math.sin(frame * 0.71 + 0.4) + 0.1 * Math.sin(frame * 1.9 + 3));
  const r = 0.25 * Math.sin(frame * 0.19 + 0.7);
  return {x, y, r};
};

export const HandheldShake: React.FC<{
  children: React.ReactNode;
  amp?: number;
  zoom?: number;
  panX?: number;
  panY?: number;
  rotate?: number;
  /** transform-origin, e.g. "50% 40%" */
  origin?: string;
}> = ({children, amp = 3, zoom = 1, panX = 0, panY = 0, rotate = 0, origin = '50% 45%'}) => {
  const frame = useCurrentFrame();
  const s = shakeAt(frame, amp);
  return (
    <AbsoluteFill
      style={{
        transform: `translate(${s.x + panX}px, ${s.y + panY}px) rotate(${s.r + rotate}deg) scale(${zoom})`,
        transformOrigin: origin,
      }}
    >
      {children}
    </AbsoluteFill>
  );
};

/** 2.5D parallax layer: depth < 1 moves/zooms less (background), > 1 more (foreground). */
export const DepthLayer: React.FC<{children: React.ReactNode; zoom: number; panX?: number; depth: number}> = ({
  children,
  zoom,
  panX = 0,
  depth,
}) => (
  <AbsoluteFill
    style={{
      transform: `translateX(${panX * depth}px) scale(${1 + (zoom - 1) * depth})`,
      transformOrigin: '50% 42%',
    }}
  >
    {children}
  </AbsoluteFill>
);
