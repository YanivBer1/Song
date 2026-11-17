import React from 'react';
import {AbsoluteFill, interpolate, random, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {CharacterYaniv, Expression} from '../components/CharacterYaniv';
import {DoodleArrow} from '../components/DoodleArrow';
import {Burst} from '../components/Doodles';
import {HandheldShake} from '../components/HandheldShake';
import {KineticText} from '../components/KineticText';
import {PhoneMockup} from '../components/PhoneMockup';
import {Room} from '../components/Room';
import {BOUNCE, C, HEEBO, POP, SAFE} from '../theme';
import {MouzaAppScreen, PHONE, SCREEN_W} from './Scene3Solution';

// SCENE 4 – RETAKE – frames 270–390 (9.0–13.0 s), local 0–120
// Teleprompter. Yaniv trips on ONE word → red underline. Arrow → he taps
// "צלם שוב רק את הקטע הזה" → only that one segment re-records. Not the video.

export const MISTAKE_AT = 22;
export const RETAKE_TAP = 62;
const RETAKE_DONE = 100;
const MISTAKE_WORD = 5;

const LINE = 'כל מוצר בחנות עובר דרך הידיים שלי.'.split(' ');
const FIRST_CUES = [4, 8, 12, 16, 19, MISTAKE_AT];
const RETAKE_CUES = [72, 78, 84, 88, 92, 96, 102];

const Teleprompter: React.FC<{frame: number}> = ({frame}) => {
  const {fps} = useVideoConfig();
  const retaking = frame >= RETAKE_TAP + 6;
  const cues = retaking ? RETAKE_CUES : FIRST_CUES;
  let cur = -1;
  cues.forEach((c, i) => {
    if (frame >= c) cur = i;
  });
  const mistake = frame >= MISTAKE_AT && !retaking;
  const underline = spring({frame: frame - MISTAKE_AT, fps, config: {damping: 16, stiffness: 120}});
  return (
    <div
      dir="rtl"
      style={{
        position: 'absolute',
        top: 424,
        left: 20,
        width: SCREEN_W - 40,
        padding: '18px 22px',
        background: 'rgba(0,0,0,0.78)',
        borderRadius: 24,
        display: 'flex',
        flexWrap: 'wrap',
        columnGap: 12,
        rowGap: 4,
        direction: 'rtl',
        justifyContent: 'flex-start',
        fontFamily: HEEBO,
        fontWeight: 700,
        fontSize: 46,
        lineHeight: 1.25,
      }}
    >
      {LINE.map((w, i) => {
        const isMistake = i === MISTAKE_WORD && mistake;
        const fixed = retaking && i >= MISTAKE_WORD && frame >= RETAKE_CUES[i];
        const color = isMistake ? C.red : i === cur ? C.yellow : i < cur ? (fixed ? C.teal : C.white) : C.gray;
        const jitter = isMistake && frame < MISTAKE_AT + 10 ? Math.sin(frame * 3) * 5 : 0;
        return (
          <span
            key={i}
            dir="rtl"
            style={{
              position: 'relative',
              display: 'inline-block',
              color,
              opacity: i > cur && !isMistake ? 0.4 : 1,
              transform: `translateX(${jitter}px) scale(${i === cur || isMistake ? 1.08 : 1})`,
              unicodeBidi: 'isolate',
            }}
          >
            {w}
            {isMistake ? (
              <svg viewBox="0 0 100 20" preserveAspectRatio="none" style={{position: 'absolute', left: -6, right: -6, bottom: -10, width: 'calc(100% + 12px)', height: 18, overflow: 'visible'}}>
                <path
                  d={`M 0 10 Q 12 ${4 + random(`u${Math.floor(frame / 4)}`) * 6} 25 10 T 50 10 T 75 10 T 100 9`}
                  stroke={C.red}
                  strokeWidth={6}
                  fill="none"
                  strokeLinecap="round"
                  vectorEffect="non-scaling-stroke"
                  pathLength={1}
                  strokeDasharray={1}
                  strokeDashoffset={1 - underline}
                />
              </svg>
            ) : null}
          </span>
        );
      })}
    </div>
  );
};

const Timeline: React.FC<{frame: number}> = ({frame}) => {
  const {fps} = useVideoConfig();
  const inS = spring({frame: frame - 30, fps, config: BOUNCE});
  const recording = frame >= RETAKE_TAP + 4 && frame < RETAKE_DONE;
  const fill = interpolate(frame, [RETAKE_TAP + 4, RETAKE_DONE], [0, 1], {extrapolateLeft: 'clamp', extrapolateRight: 'clamp'});
  const done = frame >= RETAKE_DONE;
  const doneS = spring({frame: frame - RETAKE_DONE, fps, config: POP});
  if (frame < 28) return null;
  return (
    <div dir="rtl" style={{position: 'absolute', top: 628, left: 24, width: SCREEN_W - 48, height: 48, display: 'flex', gap: 8, direction: 'rtl', transform: `scaleX(${inS})`}}>
      {[0, 1, 2, 3, 4].map((i) => {
        const bad = i === 2;
        const pulse = bad && recording ? 1 + 0.12 * Math.abs(Math.sin(frame / 3)) : bad && done ? 1 + 0.25 * (1 - doneS) : 1;
        return (
          <div
            key={i}
            style={{
              flex: 1,
              borderRadius: 10,
              position: 'relative',
              overflow: 'hidden',
              background: bad ? (done ? C.teal : '#4A1F22') : '#3A3F4B',
              border: `4px solid ${bad ? (done ? C.teal : C.red) : '#5A606C'}`,
              transform: `scale(${pulse})`,
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            {bad && recording ? <div style={{position: 'absolute', right: 0, top: 0, bottom: 0, width: `${fill * 100}%`, background: C.teal}} /> : null}
            {bad && done ? (
              <svg width={36} height={30} viewBox="0 0 36 30" style={{position: 'relative', transform: `scale(${doneS})`}}>
                <path d="M 4 16 L 14 26 L 32 4" stroke={C.bg} strokeWidth={7} fill="none" strokeLinecap="round" strokeLinejoin="round" />
              </svg>
            ) : null}
          </div>
        );
      })}
    </div>
  );
};

const RetakeButton: React.FC<{frame: number}> = ({frame}) => {
  const {fps} = useVideoConfig();
  const inS = spring({frame: frame - 34, fps, config: POP});
  const tapped = frame >= RETAKE_TAP;
  const tapS = spring({frame: frame - RETAKE_TAP, fps, config: POP});
  const pulse = !tapped && frame > 40 ? 1 + 0.05 * Math.sin(frame / 2.2) : 1;
  const press = tapped && frame < RETAKE_TAP + 5 ? 0.88 : 1;
  if (frame < 34) return null;
  return (
    <div
      dir="rtl"
      style={{
        position: 'absolute',
        top: 694,
        left: 26,
        width: SCREEN_W - 52,
        height: 92,
        borderRadius: 46,
        background: tapped ? C.teal : C.red,
        border: `5px solid ${C.bg}`,
        boxShadow: `0 7px 0 ${C.bg}`,
        color: tapped ? C.bg : C.white,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        gap: 12,
        fontFamily: HEEBO,
        fontWeight: 800,
        fontSize: 34,
        transform: `scale(${inS * pulse * press * (tapped ? 1 + 0.08 * (1 - tapS) : 1)})`,
        whiteSpace: 'nowrap',
      }}
    >
      <span style={{fontSize: 40}}>↻</span>
      <span dir="rtl">צלם שוב רק את הקטע הזה</span>
    </div>
  );
};

const CameraScreen: React.FC<{frame: number}> = ({frame}) => {
  const retaking = frame >= RETAKE_TAP + 6 && frame < RETAKE_DONE + 6;
  let expr: Expression = 'neutral';
  if (frame >= MISTAKE_AT && frame < 40) expr = 'wince';
  else if (frame >= 40 && frame < RETAKE_TAP + 6) expr = 'embarrassed';
  else if (frame >= RETAKE_DONE) expr = 'smile';
  const talking = (frame < MISTAKE_AT && frame > 3) || retaking;
  const mouth = talking ? 0.2 + 0.7 * random(`mini${Math.floor(frame / 2)}`) : 0;
  const secs = Math.floor(frame / 30) + 4;
  return (
    <AbsoluteFill style={{background: '#101114'}}>
      {/* rec bar */}
      <div dir="rtl" style={{position: 'absolute', top: 72, right: 34, display: 'flex', alignItems: 'center', gap: 12, fontFamily: HEEBO, fontWeight: 700, fontSize: 32, color: C.white}}>
        <div style={{width: 22, height: 22, borderRadius: 11, background: C.red, opacity: frame % 20 < 12 ? 1 : 0.3}} />
        <span dir="rtl">{`00:0${secs % 10}`}</span>
      </div>
      {/* front-camera preview */}
      <div style={{position: 'absolute', top: 120, left: 20, width: SCREEN_W - 40, height: 296, borderRadius: 26, overflow: 'hidden', border: `4px solid #2A2D35`}}>
        <div style={{position: 'absolute', left: -280, top: -500, width: 1080, height: 1920, transform: 'scale(0.55)'}}>
          <Room blur={4} />
        </div>
        <div style={{position: 'absolute', left: 58, top: -64}}>
          <CharacterYaniv width={420} expression={expr} mouthOpen={mouth} tilt={frame >= MISTAKE_AT && frame < 32 ? Math.sin(frame * 2) * 4 : 0} />
        </div>
        {/* "only this part" badge while retaking */}
        {retaking ? (
          <div style={{position: 'absolute', inset: 0, border: `8px solid ${C.teal}`, borderRadius: 22, opacity: frame % 10 < 6 ? 1 : 0.4}} />
        ) : null}
      </div>
      <Teleprompter frame={frame} />
      <Timeline frame={frame} />
      <RetakeButton frame={frame} />
    </AbsoluteFill>
  );
};

const Finger: React.FC<{x: number; y: number}> = ({x, y}) => (
  <svg width={260} height={700} viewBox="0 0 260 700" style={{position: 'absolute', left: x - 50, top: y - 40, overflow: 'visible'}}>
    <path d="M 120 700 L 140 330 L 250 330 L 260 700 Z" fill={C.shirt} stroke={C.ink} strokeWidth={8} />
    <path d="M 40 60 Q 60 0 100 30 L 150 290 L 210 260 Q 250 300 240 360 L 150 380 Q 100 360 90 300 Z" fill={C.skin} stroke={C.ink} strokeWidth={8} strokeLinejoin="round" />
    <ellipse cx={66} cy={44} rx={20} ry={16} fill="#F7D6C2" transform="rotate(-20 66 44)" />
  </svg>
);

export const Scene4Retake: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const swap = spring({frame: frame + 1, fps, config: POP});
  const shake = frame >= MISTAKE_AT && frame < MISTAKE_AT + 8 ? Math.sin(frame * 4) * 2.5 : 0;
  // finger comes in for the tap, then leaves
  const fingerIn = spring({frame: frame - (RETAKE_TAP - 10), fps, config: POP});
  const fingerOut = spring({frame: frame - (RETAKE_TAP + 8), fps, config: POP});
  const fy = 1150 + (1 - fingerIn) * 700 + fingerOut * 700;
  const btnY = PHONE.y + 22 + 694 + 46;

  return (
    <AbsoluteFill>
      <HandheldShake amp={3}>
        <div style={{position: 'absolute', inset: 0, transform: 'scale(1.25)'}}>
          <Room blur={10} dim={0.25} />
        </div>
        <PhoneMockup x={PHONE.x} y={PHONE.y} width={PHONE.w} height={PHONE.h} rotate={-2 + shake}>
          <MouzaAppScreen frame={105} />
          <div style={{position: 'absolute', inset: 0, transform: `translateX(${(1 - swap) * -110}%)`}}>
            <CameraScreen frame={frame} />
          </div>
        </PhoneMockup>
        <svg viewBox="0 0 1080 1920" width={1080} height={1920} style={{position: 'absolute', inset: 0, overflow: 'visible'}}>
          <path d="M 120 1920 Q 150 1640 236 1520" stroke={C.ink} strokeWidth={130} fill="none" strokeLinecap="round" />
          <path d="M 120 1920 Q 150 1640 236 1520" stroke={C.skin} strokeWidth={112} fill="none" strokeLinecap="round" />
          {[1240, 1320, 1400].map((y) => (
            <ellipse key={y} cx={238} cy={y} rx={34} ry={30} fill={C.skin} stroke={C.ink} strokeWidth={7} />
          ))}
        </svg>
        {frame >= RETAKE_TAP - 12 && frame < RETAKE_TAP + 30 ? <Finger x={560} y={fy} /> : null}
      </HandheldShake>
      <DoodleArrow start={40} from={[120, 760]} to={[262, btnY - 10]} bend={-90} color={C.yellow} />
      <Burst start={RETAKE_TAP} cx={540} cy={btnY} r={150} color={C.teal} rays={9} />
      <KineticText
        text="טעית? מצלמים רק את הקטע הזה."
        cues={[4, 36, 80, 86, 92, 98]}
        appearAt={1}
        top={SAFE.top + 2}
        fontSize={62}
        tilt={-1}
        holdLast={18}
        circle={{word: 2, at: 84, color: C.teal}}
      />
    </AbsoluteFill>
  );
};
