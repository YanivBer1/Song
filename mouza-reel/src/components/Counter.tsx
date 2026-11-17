import React from 'react';
import {random, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {C, HEEBO, POP} from '../theme';

// Sticker counter for the hook: "ניסיון 1" → "ניסיון 7" → "ניסיון 11".
// Every new value slams in with an overshoot spring + punch zoom + a
// new random tilt, like a creator slapping stickers on in CapCut.

export type CounterIcon = 'rec' | 'chat' | 'trash' | 'receipt' | 'calendar';
export type CounterItem = {label: string; at: number};

const Icon: React.FC<{icon: CounterIcon; frame: number}> = ({icon, frame}) => {
  const s = 64;
  switch (icon) {
    case 'rec':
      return (
        <svg width={s} height={s} viewBox="0 0 64 64">
          <circle cx={32} cy={32} r={22} fill={C.red} opacity={frame % 20 < 12 ? 1 : 0.35} />
          <circle cx={32} cy={32} r={28} fill="none" stroke={C.red} strokeWidth={4} />
        </svg>
      );
    case 'chat':
      return (
        <svg width={s} height={s} viewBox="0 0 64 64">
          <path d="M 8 12 H 56 V 42 H 26 L 14 54 V 42 H 8 Z" fill={C.teal} stroke={C.ink} strokeWidth={4} strokeLinejoin="round" />
          <circle cx={22} cy={27} r={4} fill={C.ink} />
          <circle cx={32} cy={27} r={4} fill={C.ink} />
          <circle cx={42} cy={27} r={4} fill={C.ink} />
        </svg>
      );
    case 'trash':
      return (
        <svg width={s} height={s} viewBox="0 0 64 64">
          <path d="M 14 18 H 50 L 46 58 H 18 Z" fill={C.gray} stroke={C.ink} strokeWidth={4} strokeLinejoin="round" />
          <path d="M 10 16 H 54 M 26 10 H 38" stroke={C.ink} strokeWidth={5} strokeLinecap="round" />
          <path d="M 26 26 V 50 M 38 26 V 50" stroke={C.ink} strokeWidth={4} />
        </svg>
      );
    case 'receipt':
      return (
        <svg width={s} height={s} viewBox="0 0 64 64">
          <path d="M 14 6 H 50 V 58 L 44 53 L 38 58 L 32 53 L 26 58 L 20 53 L 14 58 Z" fill="#fff" stroke={C.ink} strokeWidth={4} strokeLinejoin="round" />
          <path d="M 22 20 H 42 M 22 30 H 42 M 22 40 H 34" stroke={C.red} strokeWidth={4} strokeLinecap="round" />
        </svg>
      );
    case 'calendar':
      return (
        <svg width={s} height={s} viewBox="0 0 64 64">
          <rect x={8} y={12} width={48} height={44} rx={6} fill="#fff" stroke={C.ink} strokeWidth={4} />
          <rect x={8} y={12} width={48} height={14} rx={6} fill={C.red} stroke={C.ink} strokeWidth={4} />
          <path d="M 20 6 V 18 M 44 6 V 18" stroke={C.ink} strokeWidth={5} strokeLinecap="round" />
          <path d="M 22 36 L 42 50 M 42 36 L 22 50" stroke={C.ink} strokeWidth={4} strokeLinecap="round" />
        </svg>
      );
  }
};

export const Counter: React.FC<{
  items: CounterItem[];
  icon: CounterIcon;
  /** center point of the sticker in frame space */
  x: number;
  y: number;
  /** hide after this frame */
  until?: number;
}> = ({items, icon, x, y, until}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  let idx = -1;
  items.forEach((it, i) => {
    if (frame >= it.at) idx = i;
  });
  if (idx < 0 || (until !== undefined && frame > until)) return null;
  const item = items[idx];
  // the very first sticker is already mid-pop on frame 0 (no empty first frame)
  const local = frame - item.at + (idx === 0 ? 4 : 0);
  const s = spring({frame: local, fps, config: POP});
  const scale = 0.4 + 0.6 * s + (1 - s) * 0.5;
  const tilt = (random(`cnt${idx}${item.label}`) - 0.5) * 14;
  const tokens = item.label.split(' ');

  return (
    <div
      dir="rtl"
      style={{
        position: 'absolute',
        left: x,
        top: y,
        transform: `translate(-50%, -50%) scale(${scale}) rotate(${tilt}deg)`,
        display: 'flex',
        alignItems: 'center',
        gap: 16,
        direction: 'rtl',
        background: C.white,
        color: C.bg,
        padding: '10px 30px 14px 30px',
        borderRadius: 22,
        border: `5px solid ${C.bg}`,
        boxShadow: `8px 10px 0 ${C.bg}`,
        fontFamily: HEEBO,
        fontWeight: 800,
        fontSize: 76,
        lineHeight: 1,
        whiteSpace: 'nowrap',
      }}
    >
      <Icon icon={icon} frame={frame} />
      <span dir="rtl" style={{display: 'flex', gap: 18, alignItems: 'baseline', unicodeBidi: 'isolate'}}>
        {tokens.map((t, i) => (
          <span key={i} style={{color: /\d/.test(t) ? C.red : C.bg, fontSize: /\d/.test(t) ? 92 : 76}}>
            {t}
          </span>
        ))}
      </span>
    </div>
  );
};
