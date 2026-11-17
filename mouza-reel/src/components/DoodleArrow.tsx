import React from 'react';
import {spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {C} from '../theme';
import {boilKey, useDraw, wobblyPath} from './Doodles';

// Hand-drawn marker arrow, drawn on in real time (stroke-dashoffset),
// with a springy arrowhead that "snaps" on at the end.
// Coordinates are in 1080×1920 frame space.

export type DoodleArrowProps = {
  start: number;
  from: [number, number];
  to: [number, number];
  /** perpendicular bend of the curve in px (+/−) */
  bend?: number;
  color?: string;
  width?: number;
  /** wiggle the arrow back and forth after it lands (pointing gesture) */
  nudge?: boolean;
};

export const DoodleArrow: React.FC<DoodleArrowProps> = ({
  start,
  from,
  to,
  bend = 120,
  color = C.yellow,
  width = 14,
  nudge = true,
}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const p = useDraw(start, 1.3);
  if (frame < start) return null;

  // after landing, the whole arrow pokes toward the target a few times
  const poke = nudge && p > 0.95 ? Math.sin((frame - start) * 0.55) * 10 : 0;
  const [fx, fy] = from;
  const dx0 = to[0] - fx;
  const dy0 = to[1] - fy;
  const len0 = Math.hypot(dx0, dy0) || 1;
  const ux = dx0 / len0;
  const uy = dy0 / len0;
  const tx = to[0] + ux * poke;
  const ty = to[1] + uy * poke;

  const mx = (fx + tx) / 2 - uy * bend;
  const my = (fy + ty) / 2 + ux * bend;

  // sample the quadratic so we can jitter it like a marker line
  const pts: [number, number][] = [];
  const N = 10;
  for (let i = 0; i <= N; i++) {
    const t = i / N;
    pts.push([
      (1 - t) * (1 - t) * fx + 2 * (1 - t) * t * mx + t * t * tx,
      (1 - t) * (1 - t) * fy + 2 * (1 - t) * t * my + t * t * ty,
    ]);
  }
  const k = boilKey(frame);
  const d = wobblyPath(pts, 7, `arr${start}${k}`);

  // arrowhead direction = tangent at the end
  const ex = tx - mx;
  const ey = ty - my;
  const el = Math.hypot(ex, ey) || 1;
  const hx = ex / el;
  const hy = ey / el;
  const headLen = 54;
  const spread = 0.55;
  const rot = (a: number): [number, number] => [
    tx - headLen * (hx * Math.cos(a) - hy * Math.sin(a)),
    ty - headLen * (hy * Math.cos(a) + hx * Math.sin(a)),
  ];
  const [h1x, h1y] = rot(spread);
  const [h2x, h2y] = rot(-spread);
  const headPop = spring({frame: frame - start - 12, fps, config: {damping: 8, stiffness: 260, mass: 0.5}});

  return (
    <svg
      viewBox="0 0 1080 1920"
      width={1080}
      height={1920}
      style={{position: 'absolute', left: 0, top: 0, overflow: 'visible', pointerEvents: 'none'}}
    >
      {/* dark under-stroke so the arrow reads on any background */}
      <path d={d} fill="none" stroke={C.bg} strokeOpacity={0.55} strokeWidth={width + 10} strokeLinecap="round" pathLength={1} strokeDasharray={1} strokeDashoffset={1 - p} />
      <path d={d} fill="none" stroke={color} strokeWidth={width} strokeLinecap="round" pathLength={1} strokeDasharray={1} strokeDashoffset={1 - p} />
      {p > 0.7 ? (
        <g transform={`translate(${tx} ${ty}) scale(${headPop}) translate(${-tx} ${-ty})`}>
          <path
            d={`M ${h1x} ${h1y} L ${tx} ${ty} L ${h2x} ${h2y}`}
            fill="none"
            stroke={color}
            strokeWidth={width}
            strokeLinecap="round"
            strokeLinejoin="round"
          />
        </g>
      ) : null}
    </svg>
  );
};
