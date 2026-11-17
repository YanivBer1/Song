import React from 'react';
import {useCurrentFrame} from 'remotion';
import {C, HEEBO} from '../theme';

// Yaniv's messy home-office. Same room behind him in every scene.
// Easter eggs that earn rewatches: a ring light covered in cobwebs (never
// used) and a sticky note that says "מחר" (tomorrow).

export const Room: React.FC<{blur?: number; dim?: number}> = ({blur = 3, dim = 0}) => {
  const frame = useCurrentFrame();
  const flicker = 0.9 + 0.1 * Math.sin(frame / 3.3) * Math.sin(frame / 7.1);
  return (
    <div style={{position: 'absolute', inset: 0, filter: `blur(${blur}px)`}}>
      <svg viewBox="0 0 1080 1920" width="100%" height="100%" preserveAspectRatio="xMidYMid slice">
        <defs>
          <linearGradient id="wallG" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0" stopColor={C.wallDark} />
            <stop offset="0.55" stopColor={C.wall} />
            <stop offset="1" stopColor={C.wallDark} />
          </linearGradient>
          <radialGradient id="lampG" cx="0.5" cy="0.5" r="0.5">
            <stop offset="0" stopColor={C.yellow} stopOpacity={0.35} />
            <stop offset="1" stopColor={C.yellow} stopOpacity={0} />
          </radialGradient>
          <radialGradient id="vign" cx="0.5" cy="0.45" r="0.75">
            <stop offset="0.55" stopColor="#000" stopOpacity={0} />
            <stop offset="1" stopColor="#000" stopOpacity={0.6} />
          </radialGradient>
        </defs>

        {/* walls – 2.5D corner */}
        <rect width={1080} height={1920} fill="url(#wallG)" />
        <path d="M 790 0 L 1080 0 L 1080 1920 L 790 1920 Z" fill={C.wallDark} opacity={0.65} />
        <path d="M 790 0 L 786 1920" stroke="#15171D" strokeWidth={6} />

        {/* window with night outside */}
        <g transform="rotate(-1.5 240 560)">
          <rect x={70} y={360} width={330} height={400} rx={10} fill="#141A26" stroke="#3B4150" strokeWidth={14} />
          <path d="M 235 360 L 235 760 M 70 560 L 400 560" stroke="#3B4150" strokeWidth={10} />
          <circle cx={330} cy={430} r={28} fill="#E8E2C8" opacity={0.7} />
          <path d="M 90 740 L 150 650 L 190 700 L 240 610 L 300 700 L 380 640 L 380 740 Z" fill="#0F131C" />
        </g>

        {/* wonky poster */}
        <g transform="rotate(5 560 380)">
          <rect x={470} y={270} width={200} height={250} fill="#2F3440" stroke="#454B59" strokeWidth={8} />
          <circle cx={570} cy={360} r={40} fill={C.yellow} opacity={0.75} />
          <path d="M 490 500 L 540 420 L 580 470 L 620 410 L 650 500 Z" fill={C.teal} opacity={0.6} />
        </g>

        {/* ring light with cobwebs (never used) */}
        <g opacity={0.95}>
          <path d="M 940 520 L 940 1500 M 940 1500 L 880 1640 M 940 1500 L 1000 1640" stroke="#3A3F4B" strokeWidth={12} />
          <circle cx={940} cy={420} r={110} fill="none" stroke="#4A505E" strokeWidth={34} />
          <circle cx={940} cy={420} r={110} fill="none" stroke="#5C6372" strokeWidth={6} />
          <g stroke="#9AA0AA" strokeWidth={2} opacity={0.75} fill="none">
            <path d="M 850 360 L 900 420 L 870 470 M 900 420 L 940 330 M 900 420 L 960 440" />
            <path d="M 878 395 Q 890 410 912 400 Q 920 420 905 440 Q 890 445 882 432" />
            <path d="M 866 380 Q 890 405 925 380 Q 945 425 915 458 Q 885 470 870 450" />
          </g>
        </g>

        {/* shelf with books and a mug */}
        <g transform="rotate(-1 600 820)">
          <rect x={460} y={820} width={300} height={18} fill="#4A3D35" />
          <rect x={480} y={712} width={34} height={108} fill={C.teal} opacity={0.8} />
          <rect x={518} y={730} width={28} height={90} fill={C.red} opacity={0.75} />
          <rect x={550} y={700} width={40} height={120} fill={C.yellow} opacity={0.75} />
          <rect x={600} y={760} width={30} height={64} fill={C.gray} transform="rotate(-22 615 820)" />
          <rect x={660} y={770} width={58} height={50} rx={8} fill="#E9E4D8" />
          <path d="M 718 785 q 22 0 22 18 q 0 18 -22 16" fill="none" stroke="#E9E4D8" strokeWidth={8} />
        </g>

        {/* sticky note: "מחר" */}
        <g transform="rotate(-8 160 1040)">
          <rect x={90} y={980} width={150} height={140} fill={C.yellow} />
          <path d="M 90 980 L 240 980 L 240 996 L 90 996 Z" fill="#E6BC32" />
          <text
            x={165}
            y={1075}
            textAnchor="middle"
            direction="rtl"
            fontFamily={HEEBO}
            fontWeight={800}
            fontSize={56}
            fill={C.red}
          >
            מחר
          </text>
        </g>

        {/* plant */}
        <g>
          <path d="M 90 1300 L 250 1300 L 230 1460 L 110 1460 Z" fill="#B56A4A" />
          <path d="M 170 1300 Q 120 1180 60 1150 Q 130 1170 168 1260 Q 160 1150 200 1080 Q 196 1180 182 1290 Q 230 1180 290 1160 Q 230 1220 190 1300" fill="#2E7D5B" />
        </g>

        {/* warm desk-lamp glow */}
        <circle cx={130} cy={1600} r={420} fill="url(#lampG)" opacity={flicker} />

        <rect width={1080} height={1920} fill="url(#vign)" />
        {dim > 0 ? <rect width={1080} height={1920} fill="#000" opacity={dim} /> : null}
      </svg>
    </div>
  );
};
