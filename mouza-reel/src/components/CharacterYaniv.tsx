import React, {useId} from 'react';
import {random, useCurrentFrame} from 'remotion';
import {C} from '../theme';

// ============================================================================
//  YANIV – the brand character. One design, reused in every scene.
//  Late 30s, short beard, black tee, messy hair, tired eyes, big face.
//  Drawn as flat 2.5D doodle vector with a "line boil" (hand-drawn wobble
//  that re-rolls every 4 frames) so he never looks like a clean corporate
//  render. Do NOT redesign him per scene – only change props.
// ============================================================================

export type Expression =
  | 'neutral'
  | 'embarrassed'
  | 'laugh'
  | 'sad'
  | 'surprised'
  | 'smile'
  | 'frozen'
  | 'wince'
  | 'smirk'
  | 'proud';

type MouthType = 'line' | 'wobbly' | 'o' | 'smirk' | 'grit';

type ExprDef = {
  browTilt: number; // + = worried (inner ends up), − = angry
  browY: number; // − = raised
  browAsym?: number; // raises character's left brow (viewer's right) only
  lid: number; // 0 = wide open, 1 = closed (Yaniv is always a bit tired)
  lowerLid?: number; // squint from below
  pupil: number; // pupil scale
  mouth: MouthType;
  curv: number; // + smile, − frown
  baseOpen: number; // mouth opening without speech
  eyesClosed?: boolean; // happy "^ ^" eyes
  blush: number;
  sweat: number;
  look?: [number, number];
};

const EXPR: Record<Expression, ExprDef> = {
  neutral: {browTilt: 0, browY: 0, lid: 0.34, pupil: 1, mouth: 'line', curv: 0.1, baseOpen: 0, blush: 0, sweat: 0},
  embarrassed: {
    browTilt: 14,
    browY: -12,
    lid: 0.36,
    pupil: 0.9,
    mouth: 'wobbly',
    curv: 0.2,
    baseOpen: 0,
    blush: 1,
    sweat: 1,
    look: [0.55, 0.15],
  },
  laugh: {
    browTilt: 8,
    browY: -22,
    lid: 0,
    pupil: 1,
    mouth: 'line',
    curv: 1,
    baseOpen: 0.85,
    eyesClosed: true,
    blush: 0.7,
    sweat: 0,
  },
  sad: {browTilt: 18, browY: -6, lid: 0.5, pupil: 1, mouth: 'line', curv: -0.7, baseOpen: 0, blush: 0, sweat: 0},
  surprised: {browTilt: 4, browY: -34, lid: 0.02, pupil: 0.7, mouth: 'o', curv: 0, baseOpen: 0.5, blush: 0.3, sweat: 0},
  smile: {browTilt: 2, browY: -8, lid: 0.3, pupil: 1, mouth: 'line', curv: 0.95, baseOpen: 0.05, blush: 0.25, sweat: 0},
  frozen: {
    browTilt: 6,
    browY: -28,
    lid: 0,
    pupil: 0.45,
    mouth: 'line',
    curv: -0.15,
    baseOpen: 0.08,
    blush: 0,
    sweat: 1,
  },
  wince: {
    browTilt: -10,
    browY: 8,
    lid: 0.62,
    lowerLid: 0.25,
    pupil: 1,
    mouth: 'grit',
    curv: -0.3,
    baseOpen: 0,
    blush: 0.4,
    sweat: 1,
  },
  smirk: {
    browTilt: 0,
    browY: 0,
    browAsym: -26,
    lid: 0.42,
    pupil: 1,
    mouth: 'smirk',
    curv: 0.5,
    baseOpen: 0,
    blush: 0,
    sweat: 0,
    look: [-0.1, 0],
  },
  proud: {browTilt: 4, browY: -12, lid: 0.4, pupil: 1, mouth: 'line', curv: 0.75, baseOpen: 0, blush: 0.2, sweat: 0},
};

export type YanivProps = {
  /** render width in px (height = width × 1.4) */
  width: number;
  expression?: Expression;
  /** 0..1, how open the mouth is from speech (drive it from the VO) */
  mouthOpen?: number;
  /** pupils, −1..1 each axis. Overrides the expression default */
  look?: [number, number];
  /** 0..1 shoulders drop + head sinks (scene 2) */
  slump?: number;
  /** 0..1 raises his hand and points at the camera */
  point?: number;
  /** 0..1 holds a phone at chest height and looks down at it */
  phone?: number;
  /** head tilt in degrees */
  tilt?: number;
  /** squash & stretch, 1 = rest. Feed a spring here for bouncy moves */
  squash?: number;
  /** turn line boil off (e.g. tiny thumbnails) */
  boil?: boolean;
  style?: React.CSSProperties;
};

const INK = C.ink;
const STROKE = 8;

const mouthPath = (open: number, curv: number, w = 58) => {
  const y = 668;
  const lift = curv * 16;
  const lx = 500 - w;
  const rx = 500 + w;
  const ly = y - lift;
  const topC = y + curv * 12 - open * 10;
  const botC = y + curv * 24 + open * 86;
  return `M ${lx} ${ly} Q 500 ${topC} ${rx} ${ly} Q 500 ${botC} ${lx} ${ly} Z`;
};

export const CharacterYaniv: React.FC<YanivProps> = ({
  width,
  expression = 'neutral',
  mouthOpen = 0,
  look,
  slump = 0,
  point = 0,
  phone = 0,
  tilt = 0,
  squash = 1,
  boil = true,
  style,
}) => {
  const frame = useCurrentFrame();
  const uid = useId().replace(/[^a-zA-Z0-9]/g, '');
  const e = EXPR[expression];

  // --- blink (irregular, like a real tired person) ---
  // offsets keep frame 0 and the loop frame (809) eyes-open
  const b1 = (frame + 47) % 97;
  const b2 = (frame + 90) % 151;
  const blinking = !e.eyesClosed && (b1 < 3 || b2 < 2);
  const lid = blinking ? 1 : Math.min(1, e.lid + slump * 0.12);

  // --- breathing + speech head bob ---
  const breathe = Math.sin(frame / 18) * 4;
  const speechBob = mouthOpen * 6;
  const [lx, ly] = look ?? e.look ?? [0, 0];
  const lookY = ly + phone * 0.75;
  const pupilR = 21 * e.pupil;

  // --- mouth ---
  const open = Math.max(e.baseOpen, mouthOpen);
  const talkingShape = open > 0.12;
  const curv = e.curv;
  const mouthW = expression === 'laugh' ? 74 : expression === 'surprised' ? 40 : 58;

  const boilSeed = Math.floor(frame / 4) % 9;
  const filter = boil ? `url(#boil${uid})` : undefined;

  const slumpY = slump * 46;
  const headY = slump * 34 + breathe * 0.5 + speechBob + phone * 18;

  // stubble specks – stable positions per character instance
  const stubble = Array.from({length: 46}, (_, i) => {
    const a = random(`stb-a-${i}`) * Math.PI;
    const r = 210 + random(`stb-r-${i}`) * 60;
    return {x: 500 + Math.cos(a) * r * 0.95, y: 520 + Math.sin(a) * r * 0.9};
  });

  // pointing hand travel
  const handY = 1560 - point * 470;
  const handX = 300 + point * 20;

  return (
    <svg
      viewBox="0 0 1000 1400"
      width={width}
      height={width * 1.4}
      style={{overflow: 'visible', display: 'block', ...style}}
    >
      <defs>
        <filter id={`boil${uid}`} x="-10%" y="-10%" width="120%" height="120%">
          <feTurbulence type="fractalNoise" baseFrequency="0.022" numOctaves={2} seed={boilSeed} result="n" />
          <feDisplacementMap in="SourceGraphic" in2="n" scale={7} xChannelSelector="R" yChannelSelector="G" />
        </filter>
        <clipPath id={`face${uid}`}>
          <path d="M 500 210 C 652 205 738 322 735 472 C 732 622 652 762 500 768 C 348 762 266 622 266 472 C 263 322 350 212 500 210 Z" />
        </clipPath>
        <clipPath id={`eyeL${uid}`}>
          <ellipse cx={412} cy={458} rx={64} ry={58} />
        </clipPath>
        <clipPath id={`eyeR${uid}`}>
          <ellipse cx={588} cy={458} rx={64} ry={58} />
        </clipPath>
        <clipPath id={`mouth${uid}`}>
          <path d={mouthPath(open, curv, mouthW)} />
        </clipPath>
      </defs>

      <g filter={filter}>
        <g transform={`translate(500 1400) scale(${2 - squash} ${squash}) translate(-500 -1400)`}>
          {/* ================= BODY ================= */}
          <g transform={`translate(0 ${slumpY + breathe})`}>
            {/* neck */}
            <path d="M 425 700 L 430 880 Q 500 925 570 880 L 575 700 Z" fill={C.skinShade} stroke={INK} strokeWidth={STROKE} />
            {/* black tee – shoulders drop with slump */}
            <path
              d={`M 40 1420 L 70 ${1060 + slump * 30} Q 95 ${905 + slump * 40} 300 ${862 + slump * 30} L 410 840 Q 500 900 590 840 L 700 ${
                862 + slump * 30
              } Q 905 ${905 + slump * 40} 930 ${1060 + slump * 30} L 960 1420 Z`}
              fill={C.shirt}
              stroke={INK}
              strokeWidth={STROKE}
              strokeLinejoin="round"
            />
            {/* rim light on the tee so it reads on a dark room */}
            <path
              d={`M 80 1380 L 96 ${1070 + slump * 30} Q 118 ${925 + slump * 40} 300 ${884 + slump * 30}`}
              fill="none"
              stroke={C.shirtEdge}
              strokeWidth={10}
              strokeLinecap="round"
            />
            <path
              d={`M 920 1380 L 904 ${1070 + slump * 30} Q 882 ${925 + slump * 40} 700 ${884 + slump * 30}`}
              fill="none"
              stroke={C.teal}
              strokeOpacity={0.35}
              strokeWidth={8}
              strokeLinecap="round"
            />
            {/* crew collar */}
            <path d="M 405 846 Q 500 930 595 846" fill="none" stroke={C.shirtEdge} strokeWidth={14} strokeLinecap="round" />
            {/* sleeve seams */}
            <path d={`M 205 ${1000 + slump * 35} Q 190 1100 200 1200`} fill="none" stroke={C.shirtEdge} strokeWidth={6} />
            <path d={`M 795 ${1000 + slump * 35} Q 810 1100 800 1200`} fill="none" stroke={C.shirtEdge} strokeWidth={6} />
            {/* tiny coffee stain – he is human */}
            <ellipse cx={612} cy={1110} rx={16} ry={11} fill="#2A2C33" />
          </g>

          {/* ================= HEAD ================= */}
          <g transform={`translate(0 ${headY}) rotate(${tilt + slump * -4} 500 760)`}>
            {/* ears */}
            <ellipse cx={270} cy={500} rx={36} ry={58} fill={C.skin} stroke={INK} strokeWidth={STROKE} />
            <ellipse cx={730} cy={500} rx={36} ry={58} fill={C.skin} stroke={INK} strokeWidth={STROKE} />
            <path d="M 262 480 Q 280 500 266 528" fill="none" stroke={C.skinShade} strokeWidth={7} />
            <path d="M 738 480 Q 720 500 734 528" fill="none" stroke={C.skinShade} strokeWidth={7} />

            {/* face */}
            <path
              d="M 500 210 C 652 205 738 322 735 472 C 732 622 652 762 500 768 C 348 762 266 622 266 472 C 263 322 350 212 500 210 Z"
              fill={C.skin}
            />
            <g clipPath={`url(#face${uid})`}>
              {/* flat 2.5D side shade */}
              <path d="M 640 220 C 760 300 760 620 600 780 L 800 800 L 800 200 Z" fill={C.skinShade} opacity={0.55} />
            </g>

            {/* beard */}
            <path
              d="M 268 520 C 278 650 352 800 500 806 C 648 800 722 650 732 520 C 712 600 660 640 612 646 C 572 700 428 700 388 646 C 340 640 288 600 268 520 Z"
              fill={C.hair}
              opacity={0.94}
            />
            {stubble.map((s, i) => (
              <circle key={i} cx={s.x} cy={s.y} r={3.2} fill={C.hair} opacity={0.6} />
            ))}

            <path
              d="M 500 210 C 652 205 738 322 735 472 C 732 622 652 762 500 768 C 348 762 266 622 266 472 C 263 322 350 212 500 210 Z"
              fill="none"
              stroke={INK}
              strokeWidth={STROKE}
            />

            {/* mustache */}
            <path d="M 404 628 C 440 598 560 598 596 628 C 566 620 530 632 500 622 C 470 632 434 620 404 628 Z" fill={C.hair} />

            {/* mouth */}
            {talkingShape || e.mouth === 'line' ? (
              open > 0.04 ? (
                <g>
                  <path d={mouthPath(open, curv, mouthW)} fill="#3A1214" stroke={INK} strokeWidth={STROKE} strokeLinejoin="round" />
                  <g clipPath={`url(#mouth${uid})`}>
                    <rect x={420} y={640 - curv * 20} width={160} height={22} rx={8} fill="#FFF7EC" />
                    <ellipse cx={500} cy={700 + curv * 24 + open * 70} rx={44} ry={26} fill="#C2464E" />
                  </g>
                </g>
              ) : (
                <path
                  d={`M 446 ${668 - curv * 14} Q 500 ${668 + curv * 30} 554 ${668 - curv * 14}`}
                  fill="none"
                  stroke={INK}
                  strokeWidth={STROKE}
                  strokeLinecap="round"
                />
              )
            ) : e.mouth === 'wobbly' ? (
              <path
                d="M 444 668 q 10 -12 20 0 t 20 0 t 20 0 t 20 0 t 20 0 t 12 -4"
                fill="none"
                stroke={INK}
                strokeWidth={STROKE}
                strokeLinecap="round"
                strokeLinejoin="round"
              />
            ) : e.mouth === 'o' ? (
              <ellipse cx={500} cy={676} rx={26} ry={34} fill="#3A1214" stroke={INK} strokeWidth={STROKE} />
            ) : e.mouth === 'smirk' ? (
              <path d="M 450 672 Q 520 680 566 640" fill="none" stroke={INK} strokeWidth={STROKE} strokeLinecap="round" />
            ) : (
              /* grit */
              <g>
                <rect x={446} y={650} width={108} height={36} rx={14} fill="#FFF7EC" stroke={INK} strokeWidth={STROKE} />
                <path d="M 446 668 L 554 668 M 472 652 L 472 684 M 500 652 L 500 684 M 528 652 L 528 684" stroke={INK} strokeWidth={4} />
              </g>
            )}

            {/* nose */}
            <path d="M 506 468 C 522 520 548 548 518 566 C 504 574 486 568 478 556" fill="none" stroke={INK} strokeWidth={STROKE} strokeLinecap="round" />
            <ellipse cx={498} cy={566} rx={30} ry={10} fill={C.skinShade} opacity={0.45} />

            {/* eyes */}
            {e.eyesClosed ? (
              <g>
                <path d="M 356 470 Q 412 418 468 470" fill="none" stroke={INK} strokeWidth={12} strokeLinecap="round" />
                <path d="M 532 470 Q 588 418 644 470" fill="none" stroke={INK} strokeWidth={12} strokeLinecap="round" />
              </g>
            ) : blinking ? (
              <g>
                {[412, 588].map((cx) => (
                  <g key={cx}>
                    <path d={`M ${cx - 60} 458 Q ${cx} 476 ${cx + 60} 458`} stroke={INK} strokeWidth={10} fill="none" strokeLinecap="round" />
                    <path d={`M ${cx - 52} 528 Q ${cx} 552 ${cx + 52} 528`} fill="none" stroke="#9A6656" strokeWidth={6} strokeLinecap="round" />
                  </g>
                ))}
              </g>
            ) : (
              ([
                [412, 'eyeL'],
                [588, 'eyeR'],
              ] as const).map(([cx, cid]) => (
                <g key={cid}>
                  <ellipse cx={cx} cy={458} rx={64} ry={58} fill="#FFFDF7" />
                  <g clipPath={`url(#${cid}${uid})`}>
                    <circle cx={cx + lx * 22} cy={462 + lookY * 18} r={pupilR} fill={INK} />
                    <circle cx={cx + lx * 22 + 7} cy={455 + lookY * 18} r={pupilR * 0.3} fill="#fff" />
                    {/* heavy tired upper lid */}
                    <rect x={cx - 70} y={390} width={140} height={10 + lid * 126} fill={C.skin} />
                    <path
                      d={`M ${cx - 70} ${400 + lid * 126} Q ${cx} ${392 + lid * 126} ${cx + 70} ${400 + lid * 126}`}
                      stroke={INK}
                      strokeWidth={7}
                      fill="none"
                    />
                    {e.lowerLid ? <rect x={cx - 70} y={516 - e.lowerLid * 80} width={140} height={100} fill={C.skin} /> : null}
                  </g>
                  <ellipse cx={cx} cy={458} rx={64} ry={58} fill="none" stroke={INK} strokeWidth={STROKE} />
                  {/* eye bags – the man has not slept */}
                  <path d={`M ${cx - 52} 528 Q ${cx} 552 ${cx + 52} 528`} fill="none" stroke="#9A6656" strokeWidth={6} strokeLinecap="round" />
                  <path d={`M ${cx - 32} 548 Q ${cx} 560 ${cx + 32} 548`} fill="none" stroke="#9A6656" strokeWidth={4} strokeLinecap="round" opacity={0.7} />
                </g>
              ))
            )}

            {/* brows */}
            <g transform={`translate(0 ${e.browY})`}>
              <rect
                x={350}
                y={360}
                width={124}
                height={28}
                rx={14}
                fill={C.hair}
                transform={`rotate(${-e.browTilt} 412 374)`}
              />
              <rect
                x={526}
                y={360 + (e.browAsym ?? 0)}
                width={124}
                height={28}
                rx={14}
                fill={C.hair}
                transform={`rotate(${e.browTilt + (e.browAsym ? -8 : 0)} 588 374)`}
              />
            </g>

            {/* blush */}
            {e.blush > 0 ? (
              <g opacity={0.38 * e.blush}>
                <ellipse cx={352} cy={572} rx={50} ry={24} fill={C.red} />
                <ellipse cx={648} cy={572} rx={50} ry={24} fill={C.red} />
                <path d="M 330 566 l 12 -10 M 352 566 l 12 -10 M 630 566 l 12 -10 M 652 566 l 12 -10" stroke={C.red} strokeWidth={4} />
              </g>
            ) : null}

            {/* messy hair */}
            <path
              d="M 262 420 C 236 286 300 176 420 164 C 444 118 530 110 566 150 C 612 116 712 150 706 214 C 770 236 768 342 740 420 C 724 346 694 306 652 296 C 640 330 600 314 590 280 C 556 316 498 300 478 266 C 446 312 378 312 356 288 C 326 312 296 344 262 420 Z"
              fill={C.hair}
              stroke={INK}
              strokeWidth={STROKE}
              strokeLinejoin="round"
            />
            {/* cowlick + stray strands */}
            <path d="M 520 140 C 510 90 560 70 590 96" fill="none" stroke={C.hair} strokeWidth={16} strokeLinecap="round" />
            <path d="M 470 170 C 440 120 400 130 380 112" fill="none" stroke={C.hair} strokeWidth={12} strokeLinecap="round" />
            <path d="M 690 200 C 730 170 760 180 772 160" fill="none" stroke={C.hair} strokeWidth={12} strokeLinecap="round" />
            <path d="M 330 250 Q 400 200 470 230 M 560 200 Q 620 190 670 240" fill="none" stroke="#4A3D35" strokeWidth={6} strokeLinecap="round" />

            {/* sweat drop */}
            {e.sweat > 0 ? (
              <path
                d={`M 724 ${330 + ((frame * 2) % 40)} C 704 ${370 + ((frame * 2) % 40)} 712 ${396 + ((frame * 2) % 40)} 728 ${
                  396 + ((frame * 2) % 40)
                } C 746 ${396 + ((frame * 2) % 40)} 750 ${370 + ((frame * 2) % 40)} 724 ${330 + ((frame * 2) % 40)} Z`}
                fill="#BFF3EA"
                stroke={INK}
                strokeWidth={5}
                opacity={e.sweat}
              />
            ) : null}
          </g>

          {/* ================= POINTING HAND ("you!") ================= */}
          {point > 0.01 ? (
            <g transform={`translate(0 ${slumpY})`}>
              <path d={`M 150 1500 Q 200 ${handY + 160} ${handX} ${handY + 40}`} stroke={INK} strokeWidth={104} fill="none" strokeLinecap="round" />
              <path d={`M 150 1500 Q 200 ${handY + 160} ${handX} ${handY + 40}`} stroke={C.skin} strokeWidth={88} fill="none" strokeLinecap="round" />
              <g transform={`translate(${handX} ${handY}) rotate(34)`}>
                {/* index finger aimed at the viewer, fist + tucked thumb */}
                <rect x={-22} y={-178} width={46} height={150} rx={23} fill={C.skin} stroke={INK} strokeWidth={STROKE} />
                <ellipse cx={0} cy={0} rx={74} ry={64} fill={C.skin} stroke={INK} strokeWidth={STROKE} />
                <path d="M -46 -28 Q -10 -12 36 -30 M -50 6 Q -10 22 40 4 M -46 36 Q -10 50 34 36" stroke={C.skinShade} strokeWidth={6} fill="none" />
                <ellipse cx={-48} cy={-18} rx={26} ry={40} fill={C.skin} stroke={INK} strokeWidth={STROKE} transform="rotate(30 -48 -18)" />
                <ellipse cx={2} cy={-162} rx={13} ry={10} fill="#F7D6C2" />
              </g>
              {point > 0.8 ? (
                <g stroke={C.yellow} strokeWidth={9} strokeLinecap="round" opacity={(point - 0.8) * 5}>
                  <path d={`M ${handX + 70} ${handY - 200} l 30 -40`} />
                  <path d={`M ${handX + 105} ${handY - 160} l 48 -14`} />
                  <path d={`M ${handX + 30} ${handY - 222} l 0 -48`} />
                </g>
              ) : null}
            </g>
          ) : null}

          {/* ================= PHONE IN HANDS ================= */}
          {phone > 0.01 ? (
            <g transform={`translate(0 ${(1 - phone) * 500 + slumpY})`}>
              <path d="M 230 1500 Q 300 1260 420 1180" stroke={INK} strokeWidth={100} fill="none" strokeLinecap="round" />
              <path d="M 230 1500 Q 300 1260 420 1180" stroke={C.skin} strokeWidth={84} fill="none" strokeLinecap="round" />
              <path d="M 790 1500 Q 720 1260 600 1180" stroke={INK} strokeWidth={100} fill="none" strokeLinecap="round" />
              <path d="M 790 1500 Q 720 1260 600 1180" stroke={C.skin} strokeWidth={84} fill="none" strokeLinecap="round" />
              <g transform="rotate(-6 510 1110)">
                <rect x={400} y={960} width={220} height={330} rx={34} fill="#1A1B21" stroke={INK} strokeWidth={STROKE} />
                <rect x={414} y={974} width={192} height={302} rx={24} fill={C.teal} opacity={0.85} />
                <rect x={432} y={1200} width={156} height={16} rx={8} fill={C.yellow} />
                <rect x={432} y={1226} width={110} height={16} rx={8} fill="#fff" opacity={0.8} />
              </g>
              <ellipse cx={410} cy={1170} rx={46} ry={56} fill={C.skin} stroke={INK} strokeWidth={STROKE} />
              <ellipse cx={612} cy={1170} rx={46} ry={56} fill={C.skin} stroke={INK} strokeWidth={STROKE} />
            </g>
          ) : null}
        </g>
      </g>
    </svg>
  );
};
