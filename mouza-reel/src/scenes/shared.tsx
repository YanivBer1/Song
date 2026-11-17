import React from 'react';
import {AbsoluteFill, random, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {CharacterYaniv, Expression} from '../components/CharacterYaniv';
import {DepthLayer, HandheldShake} from '../components/HandheldShake';
import {Room} from '../components/Room';
import {CaptionWordByWord} from '../components/CaptionWordByWord';
import {C, HEEBO} from '../theme';

// ---------------------------------------------------------------------------
//  The selfie close-up. Same camera, same room, same Yaniv in scenes 1, 2, 7
//  (and cropped inside scenes 5 & 6). 2.5D: room moves at 55% of the camera.
// ---------------------------------------------------------------------------
export type ShotProps = {
  zoom?: number;
  x?: number;
  rot?: number;
  expression?: Expression;
  mouth?: number;
  slump?: number;
  point?: number;
  phone?: number;
  tilt?: number;
  squash?: number;
  look?: [number, number];
  roomBlur?: number;
  roomDim?: number;
  shake?: number;
  /** vertical offset of Yaniv (px) */
  y?: number;
};

export const YanivShot: React.FC<ShotProps> = ({
  zoom = 1,
  x = -46,
  rot = 0,
  expression = 'neutral',
  mouth = 0,
  slump = 0,
  point = 0,
  phone = 0,
  tilt = 0,
  squash = 1,
  look,
  roomBlur = 3,
  roomDim = 0,
  shake = 3,
  y = 0,
}) => (
  <HandheldShake amp={shake} rotate={rot}>
    <DepthLayer zoom={zoom} panX={x} depth={0.55}>
      <Room blur={roomBlur} dim={roomDim} />
    </DepthLayer>
    <DepthLayer zoom={zoom} panX={x} depth={1}>
      <div style={{position: 'absolute', left: -110, top: 190 + y}}>
        <CharacterYaniv
          width={1300}
          expression={expression}
          mouthOpen={mouth}
          slump={slump}
          point={point}
          phone={phone}
          tilt={tilt}
          squash={squash}
          look={look}
        />
      </div>
    </DepthLayer>
  </HandheldShake>
);

/** spring that restarts at `at` – used for every pop / move */
export const usePop = (at: number, config = {damping: 9, stiffness: 220, mass: 0.6}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  return spring({frame: frame - at, fps, config});
};

/** quick white flash on hard cuts / counter slams */
export const Flash: React.FC<{at: number[]; strength?: number}> = ({at, strength = 0.35}) => {
  const frame = useCurrentFrame();
  let o = 0;
  for (const a of at) {
    const t = frame - a;
    if (t >= 0 && t < 3) o = Math.max(o, strength * (1 - t / 3));
  }
  if (o <= 0) return null;
  return <AbsoluteFill style={{background: C.white, opacity: o, mixBlendMode: 'overlay'}} />;
};

// ---------------------------------------------------------------------------
//  Lo-fi phone-camera grain over everything (re-seeded on twos)
// ---------------------------------------------------------------------------
export const Grain: React.FC<{opacity?: number}> = ({opacity = 0.07}) => {
  const frame = useCurrentFrame();
  const seed = Math.floor(frame / 2) % 12;
  return (
    <AbsoluteFill style={{pointerEvents: 'none', mixBlendMode: 'overlay', opacity}}>
      <svg width="100%" height="100%" viewBox="0 0 540 960" preserveAspectRatio="none">
        <filter id={`grain${seed}`}>
          <feTurbulence type="fractalNoise" baseFrequency="0.9" numOctaves={1} seed={seed} />
          <feColorMatrix type="saturate" values="0" />
        </filter>
        <rect width="540" height="960" filter={`url(#grain${seed})`} />
      </svg>
    </AbsoluteFill>
  );
};

// ---------------------------------------------------------------------------
//  B-ROLL (drawn in code): hands holding a product, and the shop front.
// ---------------------------------------------------------------------------
export const BrollHands: React.FC<{t: number}> = ({t}) => {
  const lift = Math.sin(t / 6) * 10 - t * 0.8;
  return (
    <AbsoluteFill style={{background: '#F3E6D3'}}>
      <svg viewBox="0 0 1080 1920" width="100%" height="100%">
        {/* wooden counter */}
        <rect x={0} y={1180} width={1080} height={740} fill="#B98559" />
        {[1240, 1330, 1450, 1600].map((y, i) => (
          <path key={i} d={`M 0 ${y} Q 540 ${y + 18} 1080 ${y - 6}`} stroke="#9A6B44" strokeWidth={6} fill="none" />
        ))}
        {/* sleeves (black tee) */}
        <path d={`M -40 1700 L 260 ${1240 + lift} L 360 ${1320 + lift} L 80 1900 Z`} fill={C.shirt} stroke={C.ink} strokeWidth={8} />
        <path d={`M 1120 1700 L 820 ${1240 + lift} L 720 ${1320 + lift} L 1000 1900 Z`} fill={C.shirt} stroke={C.ink} strokeWidth={8} />
        {/* hands */}
        <ellipse cx={330} cy={1180 + lift} rx={110} ry={90} fill={C.skin} stroke={C.ink} strokeWidth={8} />
        <ellipse cx={750} cy={1180 + lift} rx={110} ry={90} fill={C.skin} stroke={C.ink} strokeWidth={8} />
        {/* product box with ribbon */}
        <g transform={`translate(0 ${lift}) rotate(${Math.sin(t / 9) * 2} 540 1000)`}>
          <rect x={340} y={760} width={400} height={400} rx={18} fill={C.teal} stroke={C.ink} strokeWidth={10} />
          <rect x={515} y={760} width={50} height={400} fill={C.yellow} stroke={C.ink} strokeWidth={6} />
          <rect x={340} y={935} width={400} height={50} fill={C.yellow} stroke={C.ink} strokeWidth={6} />
          <path d="M 540 760 C 470 680 420 700 450 750 C 470 780 520 770 540 760 C 560 770 610 780 630 750 C 660 700 610 680 540 760 Z" fill={C.yellow} stroke={C.ink} strokeWidth={6} />
        </g>
        {/* thumbs over the box */}
        <ellipse cx={420} cy={1110 + lift} rx={40} ry={56} fill={C.skin} stroke={C.ink} strokeWidth={7} transform={`rotate(-30 420 ${1110 + lift})`} />
        <ellipse cx={660} cy={1110 + lift} rx={40} ry={56} fill={C.skin} stroke={C.ink} strokeWidth={7} transform={`rotate(30 660 ${1110 + lift})`} />
        {/* sparkles */}
        {[0, 1, 2].map((i) => {
          const a = (t / 5 + i * 2.1) % 6.28;
          const sx = 540 + Math.cos(a) * 330;
          const sy = 900 + Math.sin(a) * 260;
          return <path key={i} d={`M ${sx} ${sy - 26} L ${sx} ${sy + 26} M ${sx - 26} ${sy} L ${sx + 26} ${sy}`} stroke={C.yellow} strokeWidth={9} strokeLinecap="round" />;
        })}
      </svg>
    </AbsoluteFill>
  );
};

export const BrollShop: React.FC<{t: number}> = ({t}) => {
  const pan = -t * 2.2;
  return (
    <AbsoluteFill style={{background: '#9FD8EA', overflow: 'hidden'}}>
      <svg viewBox="0 0 1080 1920" width="100%" height="100%">
        <circle cx={860} cy={380} r={90} fill={C.yellow} />
        <g transform={`translate(${pan} 0)`}>
          {/* building */}
          <rect x={60} y={560} width={1060} height={1100} fill="#F2E3C9" stroke={C.ink} strokeWidth={10} />
          {/* striped awning */}
          {Array.from({length: 9}, (_, i) => (
            <path
              key={i}
              d={`M ${60 + i * 118} 760 L ${178 + i * 118} 760 L ${170 + i * 118} 900 Q ${119 + i * 118} 940 ${68 + i * 118} 900 Z`}
              fill={i % 2 ? C.white : C.teal}
              stroke={C.ink}
              strokeWidth={7}
            />
          ))}
          {/* window with shelves */}
          <rect x={130} y={980} width={520} height={460} fill="#2B4B57" stroke={C.ink} strokeWidth={10} />
          {[1100, 1220, 1340].map((y) => (
            <rect key={y} x={150} y={y} width={480} height={12} fill="#7A5A3E" />
          ))}
          {Array.from({length: 12}, (_, i) => (
            <rect
              key={i}
              x={170 + (i % 4) * 115}
              y={1040 + Math.floor(i / 4) * 120}
              width={70}
              height={60}
              rx={8}
              fill={[C.teal, C.yellow, C.red, '#fff'][i % 4]}
              stroke={C.ink}
              strokeWidth={5}
            />
          ))}
          {/* door */}
          <rect x={720} y={980} width={250} height={680} fill="#C9774E" stroke={C.ink} strokeWidth={10} />
          <circle cx={930} cy={1320} r={14} fill={C.yellow} stroke={C.ink} strokeWidth={4} />
          {/* "open" sign */}
          <g transform={`rotate(${Math.sin(t / 5) * 6} 845 1080)`}>
            <path d="M 845 990 L 790 1060 M 845 990 L 900 1060" stroke={C.ink} strokeWidth={5} />
            <rect x={770} y={1050} width={150} height={74} rx={10} fill={C.white} stroke={C.ink} strokeWidth={6} />
            <text x={845} y={1104} textAnchor="middle" direction="rtl" fontFamily={HEEBO} fontWeight={800} fontSize={46} fill={C.red}>
              פתוח
            </text>
          </g>
          {/* plant pots */}
          <path d="M 120 1560 L 230 1560 L 215 1660 L 135 1660 Z" fill="#B56A4A" stroke={C.ink} strokeWidth={6} />
          <circle cx={175} cy={1520} r={60} fill="#2E7D5B" stroke={C.ink} strokeWidth={6} />
        </g>
        <rect x={0} y={1660} width={1080} height={260} fill="#8A8F98" />
      </svg>
    </AbsoluteFill>
  );
};

// ---------------------------------------------------------------------------
//  "AFTER": the finished Mouza edit (used in scene 5 right panel + scene 6).
//  Rendered full 1080×1920; scene 5 shows a center crop of it.
// ---------------------------------------------------------------------------
export const AFTER_WORDS = 'רוב האנשים לא יודעים את זה עליי: כל מוצר בחנות עובר דרך הידיים שלי. בואו לראות בעצמכם.'.split(' ');

export const FinishedReel: React.FC<{
  /** frame offset into the edit (so scene 6 continues where scene 5 left off) */
  t: number;
  cues: number[];
  broll: {from: number; to: number; kind: 'hands' | 'shop'}[];
  captionTop: number;
  captionLeft: number;
  captionWidth: number;
  captionSize: number;
  mouth: number;
  expression?: Expression;
  /** where Yaniv sits inside the edit (scene 6 reframes him for the PiP) */
  yaniv?: {left: number; top: number; width: number};
}> = ({t, cues, broll, captionTop, captionLeft, captionWidth, captionSize, mouth, expression = 'smile', yaniv = {left: -110, top: 250, width: 1300}}) => {
  const active = broll.find((b) => t >= b.from && t < b.to);
  const progress = Math.min(1, t / 300);
  // jump-cut zoom changes every ~1 s like a real edit
  const beat = Math.floor(t / 28);
  const zoom = [1.0, 1.12, 1.04, 1.16, 1.06][beat % 5];
  return (
    <AbsoluteFill style={{background: C.teal, overflow: 'hidden'}}>
      {active ? (
        active.kind === 'hands' ? (
          <BrollHands t={t - active.from} />
        ) : (
          <BrollShop t={t - active.from} />
        )
      ) : (
        <AbsoluteFill style={{transform: `scale(${zoom})`, transformOrigin: '50% 40%'}}>
          {/* bright graphic backdrop */}
          <svg viewBox="0 0 1080 1920" width="100%" height="100%" style={{position: 'absolute'}}>
            <rect width={1080} height={1920} fill={C.teal} />
            <circle cx={880} cy={420} r={260} fill={C.yellow} opacity={0.9} />
            <path d="M 0 1300 Q 300 1180 540 1300 T 1080 1300 L 1080 1920 L 0 1920 Z" fill="#00B894" />
            {Array.from({length: 14}, (_, i) => (
              <circle
                key={i}
                cx={random(`dot${i}`) * 1080}
                cy={200 + random(`doty${i}`) * 1100 + Math.sin((t + i * 9) / 10) * 12}
                r={8 + random(`dr${i}`) * 10}
                fill={C.white}
                opacity={0.35}
              />
            ))}
          </svg>
          <div style={{position: 'absolute', left: yaniv.left, top: yaniv.top}}>
            <CharacterYaniv width={yaniv.width} expression={expression} mouthOpen={mouth} tilt={Math.sin(t / 14) * 2} />
          </div>
        </AbsoluteFill>
      )}
      {/* edit progress bar */}
      <div style={{position: 'absolute', top: 0, left: 0, right: 0, height: 10, background: 'rgba(0,0,0,0.25)'}}>
        <div style={{width: `${progress * 100}%`, height: '100%', background: C.yellow, marginLeft: 'auto'}} />
      </div>
      {/* music notes = "with music" */}
      {[0, 1, 2].map((i) => {
        const y = 700 - ((t * 4 + i * 120) % 380);
        return (
          <svg key={i} width={80} height={100} viewBox="0 0 80 100" style={{position: 'absolute', left: captionLeft + captionWidth - 120 - i * 90, top: y}}>
            <path d="M 30 80 L 30 15 L 70 5 L 70 65" stroke={C.ink} strokeWidth={7} fill="none" />
            <ellipse cx={20} cy={80} rx={14} ry={11} fill={C.ink} />
            <ellipse cx={60} cy={68} rx={14} ry={11} fill={C.ink} />
          </svg>
        );
      })}
      <CaptionWordByWord words={AFTER_WORDS} cues={cues} top={captionTop} left={captionLeft} width={captionWidth} fontSize={captionSize} chunk={3} />
    </AbsoluteFill>
  );
};

// ---------------------------------------------------------------------------
//  "BEFORE": the raw take. Gray, flat, Yaniv frozen in an awkward pause.
// ---------------------------------------------------------------------------
export const RawTake: React.FC<{t: number}> = ({t}) => {
  const dots = Math.floor(t / 8) % 4;
  return (
    <AbsoluteFill style={{filter: 'grayscale(1) brightness(0.72) contrast(0.85)'}}>
      <AbsoluteFill>
        <Room blur={2} />
      </AbsoluteFill>
      <div style={{position: 'absolute', left: -110, top: 250}}>
        <CharacterYaniv width={1300} expression="frozen" mouthOpen={0} boil={false} />
      </div>
      {/* awkward pause bubble */}
      <div
        style={{
          position: 'absolute',
          left: 600,
          top: 560,
          width: 150,
          height: 90,
          borderRadius: 45,
          background: '#ddd',
          border: `5px solid ${C.ink}`,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          gap: 12,
        }}
      >
        {[0, 1, 2].map((i) => (
          <div key={i} style={{width: 18, height: 18, borderRadius: 9, background: C.ink, opacity: i < dots ? 1 : 0.2}} />
        ))}
      </div>
      {/* flat waveform = silence */}
      <svg viewBox="0 0 1080 120" width={1080} height={120} style={{position: 'absolute', left: 0, top: 1330}}>
        <path
          d={`M 300 60 ${Array.from({length: 24}, (_, i) => `L ${300 + i * 20} ${60 + (random(`wf${i}${Math.floor(t / 3)}`) - 0.5) * 6}`).join(' ')}`}
          stroke="#bbb"
          strokeWidth={6}
          fill="none"
        />
      </svg>
    </AbsoluteFill>
  );
};

// ---------------------------------------------------------------------------
//  Thought bubble (scene 2)
// ---------------------------------------------------------------------------
export const ThoughtCloud: React.FC<{cx: number; cy: number; w: number; h: number; scale: number; children?: React.ReactNode}> = ({
  cx,
  cy,
  w,
  h,
  scale,
  children,
}) => {
  const frame = useCurrentFrame();
  const bumps = 11;
  const circles = Array.from({length: bumps}, (_, i) => {
    const a = (i / bumps) * Math.PI * 2;
    const wob = Math.sin(frame / 5 + i) * 3;
    return {x: Math.cos(a) * (w / 2 - 40), y: Math.sin(a) * (h / 2 - 34), r: 62 + (i % 3) * 14 + wob};
  });
  return (
    <div style={{position: 'absolute', left: cx - w / 2, top: cy - h / 2, width: w, height: h, transform: `scale(${scale})`, transformOrigin: '20% 100%'}}>
      <svg viewBox={`${-w / 2 - 60} ${-h / 2 - 60} ${w + 120} ${h + 120}`} width={w + 120} height={h + 120} style={{position: 'absolute', left: -60, top: -60, overflow: 'visible'}}>
        {circles.map((c, i) => (
          <circle key={`o${i}`} cx={c.x} cy={c.y} r={c.r + 7} fill={C.ink} />
        ))}
        <ellipse cx={0} cy={0} rx={w / 2 - 30} ry={h / 2 - 26} fill={C.ink} />
        {circles.map((c, i) => (
          <circle key={`i${i}`} cx={c.x} cy={c.y} r={c.r} fill={C.white} />
        ))}
        <ellipse cx={0} cy={0} rx={w / 2 - 36} ry={h / 2 - 32} fill={C.white} />
        {/* trailing thought dots towards Yaniv's head (down-left) */}
        <circle cx={-w / 2 + 10} cy={h / 2 + 50} r={30} fill={C.white} stroke={C.ink} strokeWidth={7} />
        <circle cx={-w / 2 - 50} cy={h / 2 + 115} r={18} fill={C.white} stroke={C.ink} strokeWidth={6} />
      </svg>
      <div style={{position: 'absolute', inset: 0}}>{children}</div>
    </div>
  );
};

// ---------------------------------------------------------------------------
//  Paper plane (scene 7 "send")
// ---------------------------------------------------------------------------
export const PaperPlane: React.FC<{x: number; y: number; rot: number; s?: number}> = ({x, y, rot, s = 1}) => (
  <svg width={160} height={120} viewBox="0 0 160 120" style={{position: 'absolute', left: x - 80, top: y - 60, transform: `rotate(${rot}deg) scale(${s})`, overflow: 'visible'}}>
    <path d="M 6 60 L 154 8 L 104 112 L 78 72 Z" fill={C.white} stroke={C.ink} strokeWidth={7} strokeLinejoin="round" />
    <path d="M 154 8 L 78 72 L 70 104 L 60 66" fill="#DDE3EA" stroke={C.ink} strokeWidth={6} strokeLinejoin="round" />
  </svg>
);
