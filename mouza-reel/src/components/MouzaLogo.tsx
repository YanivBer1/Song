import React from 'react';
import {C, HEEBO} from '../theme';

// Mouza mark + wordmark, drawn in code. Small and late – never a logo wall.

export const MouzaMark: React.FC<{size: number}> = ({size}) => (
  <svg width={size} height={size} viewBox="0 0 100 100" style={{display: 'block'}}>
    <rect x={4} y={4} width={92} height={92} rx={26} fill={C.teal} stroke={C.ink} strokeWidth={5} />
    {/* play-button speech bubble */}
    <path d="M 30 26 H 74 Q 80 26 80 32 V 62 Q 80 68 74 68 H 46 L 32 80 V 68 H 30 Q 24 68 24 62 V 32 Q 24 26 30 26 Z" fill={C.white} />
    <path d="M 46 36 L 62 47 L 46 58 Z" fill={C.ink} />
    <circle cx={80} cy={22} r={9} fill={C.yellow} stroke={C.ink} strokeWidth={4} />
  </svg>
);

export const MouzaWordmark: React.FC<{height: number; color?: string}> = ({height, color = C.white}) => (
  <div dir="rtl" style={{display: 'flex', alignItems: 'center', gap: height * 0.25, direction: 'rtl'}}>
    <MouzaMark size={height} />
    <span
      dir="rtl"
      style={{
        fontFamily: HEEBO,
        fontWeight: 800,
        fontSize: height * 0.82,
        color,
        lineHeight: 1,
        letterSpacing: -1,
        WebkitTextStroke: `${height * 0.06}px ${C.bg}`,
        paintOrder: 'stroke fill',
      }}
    >
      מוזה
    </span>
  </div>
);
