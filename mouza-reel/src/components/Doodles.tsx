import React from 'react';
import {interpolate, random, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {C} from '../theme';

// Hand-drawn marker doodles, all generated as SVG paths in code.
// Every doodle "re-draws" itself every 4 frames with fresh jitter
// (line boil) so it feels drawn by a person, not a vector tool.

export const boilKey = (frame: number) => Math.floor(frame / 4) % 6;

/** draw-on progress 0..1 with a springy finish */
export const useDraw = (start: number, speed = 1) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  return spring({frame: (frame - start) * speed, fps, config: {damping: 18, stiffness: 90, mass: 0.6}});
};

/** jittered polyline → smooth path */
export const wobblyPath = (pts: [number, number][], jitter: number, seed: string) => {
  const p = pts.map(([x, y], i) => [
    x + (random(`${seed}x${i}`) - 0.5) * jitter,
    y + (random(`${seed}y${i}`) - 0.5) * jitter,
  ]);
  let d = `M ${p[0][0]} ${p[0][1]}`;
  for (let i = 1; i < p.length - 1; i++) {
    const mx = (p[i][0] + p[i + 1][0]) / 2;
    const my = (p[i][1] + p[i + 1][1]) / 2;
    d += ` Q ${p[i][0]} ${p[i][1]} ${mx} ${my}`;
  }
  const last = p[p.length - 1];
  d += ` L ${last[0]} ${last[1]}`;
  return d;
};

type Base = {start: number; color?: string; width?: number};

const Overlay: React.FC<{children: React.ReactNode; style?: React.CSSProperties}> = ({children, style}) => (
  <svg
    viewBox="0 0 1080 1920"
    width={1080}
    height={1920}
    style={{position: 'absolute', left: 0, top: 0, overflow: 'visible', pointerEvents: 'none', ...style}}
  >
    {children}
  </svg>
);

/** marker circle that loops ~1.15 turns like a real hand would */
export const DoodleCircle: React.FC<Base & {cx: number; cy: number; rx: number; ry: number}> = ({
  start,
  cx,
  cy,
  rx,
  ry,
  color = C.yellow,
  width = 12,
}) => {
  const frame = useCurrentFrame();
  const p = useDraw(start, 1.4);
  if (frame < start) return null;
  const k = boilKey(frame);
  const pts: [number, number][] = [];
  const turns = 1.18;
  const steps = 28;
  for (let i = 0; i <= steps; i++) {
    const a = -Math.PI * 0.75 + (i / steps) * Math.PI * 2 * turns;
    const r = 1 + (i / steps) * 0.08;
    pts.push([cx + Math.cos(a) * rx * r, cy + Math.sin(a) * ry * r]);
  }
  return (
    <Overlay>
      <path
        d={wobblyPath(pts, 10, `circ${cx}${cy}${k}`)}
        fill="none"
        stroke={color}
        strokeWidth={width}
        strokeLinecap="round"
        pathLength={1}
        strokeDasharray={1}
        strokeDashoffset={1 - p}
      />
    </Overlay>
  );
};

/** scribbly underline under a word */
export const DoodleUnderline: React.FC<Base & {x1: number; x2: number; y: number}> = ({
  start,
  x1,
  x2,
  y,
  color = C.red,
  width = 10,
}) => {
  const frame = useCurrentFrame();
  const p = useDraw(start, 1.8);
  if (frame < start) return null;
  const k = boilKey(frame);
  const pts: [number, number][] = [];
  const n = 8;
  for (let i = 0; i <= n; i++) pts.push([x1 + ((x2 - x1) * i) / n, y + Math.sin(i * 1.7) * 5]);
  // second, shorter return stroke – how people actually underline
  const back: [number, number][] = [];
  for (let i = 0; i <= n; i++) back.push([x2 - ((x2 - x1) * 0.85 * i) / n, y + 14 + Math.sin(i * 2.1) * 4]);
  return (
    <Overlay>
      <path d={wobblyPath(pts, 6, `ul${k}`)} fill="none" stroke={color} strokeWidth={width} strokeLinecap="round" pathLength={1} strokeDasharray={1} strokeDashoffset={1 - Math.min(1, p * 1.6)} />
      <path
        d={wobblyPath(back, 6, `ub${k}`)}
        fill="none"
        stroke={color}
        strokeWidth={width * 0.8}
        strokeLinecap="round"
        pathLength={1}
        strokeDasharray={1}
        strokeDashoffset={1 - Math.max(0, p * 1.6 - 0.6)}
      />
    </Overlay>
  );
};

/** big red X */
export const DoodleX: React.FC<Base & {cx: number; cy: number; size: number}> = ({start, cx, cy, size, color = C.red, width = 16}) => {
  const frame = useCurrentFrame();
  const p = useDraw(start, 2);
  if (frame < start) return null;
  const k = boilKey(frame);
  const s = size / 2;
  const a = wobblyPath(
    [
      [cx - s, cy - s],
      [cx, cy],
      [cx + s, cy + s],
    ],
    8,
    `xa${k}`,
  );
  const b = wobblyPath(
    [
      [cx + s, cy - s],
      [cx, cy],
      [cx - s, cy + s],
    ],
    8,
    `xb${k}`,
  );
  return (
    <Overlay>
      <path d={a} fill="none" stroke={color} strokeWidth={width} strokeLinecap="round" pathLength={1} strokeDasharray={1} strokeDashoffset={1 - Math.min(1, p * 2)} />
      <path d={b} fill="none" stroke={color} strokeWidth={width} strokeLinecap="round" pathLength={1} strokeDasharray={1} strokeDashoffset={1 - Math.max(0, Math.min(1, p * 2 - 1))} />
    </Overlay>
  );
};

/** little sparkle / "pop" burst lines around a point */
export const Burst: React.FC<Base & {cx: number; cy: number; r?: number; rays?: number}> = ({
  start,
  cx,
  cy,
  r = 90,
  rays = 7,
  color = C.yellow,
  width = 9,
}) => {
  const frame = useCurrentFrame();
  const t = frame - start;
  if (t < 0 || t > 14) return null;
  const grow = interpolate(t, [0, 6, 14], [0.6, 1, 1.25]);
  const fade = interpolate(t, [0, 8, 14], [1, 1, 0]);
  return (
    <Overlay>
      {Array.from({length: rays}, (_, i) => {
        const a = (i / rays) * Math.PI * 2 + random(`bst${start}${i}`) * 0.4;
        const r1 = r * grow;
        const r2 = r * grow + 34;
        return (
          <line
            key={i}
            x1={cx + Math.cos(a) * r1}
            y1={cy + Math.sin(a) * r1}
            x2={cx + Math.cos(a) * r2}
            y2={cy + Math.sin(a) * r2}
            stroke={color}
            strokeWidth={width}
            strokeLinecap="round"
            opacity={fade}
          />
        );
      })}
    </Overlay>
  );
};

/** rough rectangle outline (sketched twice, slightly off) */
export const SketchRect: React.FC<{x: number; y: number; w: number; h: number; color?: string; width?: number; seed?: string}> = ({
  x,
  y,
  w,
  h,
  color = C.white,
  width = 5,
  seed = 'r',
}) => {
  const frame = useCurrentFrame();
  const k = boilKey(frame);
  const pts = (o: number): [number, number][] => [
    [x - o, y + o],
    [x + w * 0.5, y - o],
    [x + w + o, y],
    [x + w, y + h * 0.5 + o],
    [x + w - o, y + h],
    [x + w * 0.5, y + h + o],
    [x, y + h - o],
    [x + o, y + h * 0.5],
    [x - o, y + o * 2],
  ];
  return (
    <svg style={{position: 'absolute', left: 0, top: 0, overflow: 'visible', pointerEvents: 'none'}} width={1} height={1}>
      <path d={wobblyPath(pts(3), 6, `${seed}a${k}`)} fill="none" stroke={color} strokeWidth={width} strokeLinecap="round" />
      <path d={wobblyPath(pts(-4), 6, `${seed}b${k}`)} fill="none" stroke={color} strokeWidth={width * 0.6} strokeLinecap="round" opacity={0.6} />
    </svg>
  );
};
