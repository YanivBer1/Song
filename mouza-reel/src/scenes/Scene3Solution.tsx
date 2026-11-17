import React from 'react';
import {AbsoluteFill, random, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {Burst} from '../components/Doodles';
import {KineticText} from '../components/KineticText';
import {MouzaWordmark} from '../components/MouzaLogo';
import {PhoneMockup} from '../components/PhoneMockup';
import {HandheldShake} from '../components/HandheldShake';
import {Room} from '../components/Room';
import {BOUNCE, C, HEEBO, POP, SAFE} from '../theme';

// SCENE 3 – SOLUTION – frames 165–270 (5.5–9.0 s), local 0–105
// Over-the-shoulder: Yaniv's phone, the Mouza app drawn in code.
// Finger types ONE sentence → taps "כתוב לי תסריט" → script pops line by line.

export const PHONE = {x: 230, y: 392, w: 620, h: 1300};
export const SCREEN_W = PHONE.w - 44;

export const TYPED = 'רוצה סרטון שמסביר למה לקנות אצלי';
export const SCRIPT_LINES = ['רוב האנשים לא יודעים את זה עליי:', 'כל מוצר בחנות עובר דרך הידיים שלי.', 'בואו לראות בעצמכם.'];

export const TYPE_START = 8;
export const TYPE_END = 42;
export const BUTTON_TAP = 50;
export const LINE_AT = [58, 67, 76];

const KB_ROWS = ['קראטוןםפ', 'שדגכעיחלךף', 'זסבהנמצתץ'];

const Keyboard: React.FC<{y: number; activeKey: number}> = ({y, activeKey}) => (
  <div style={{position: 'absolute', left: 0, top: y, width: SCREEN_W, height: 440, background: '#D9DCE2', paddingTop: 18}}>
    {KB_ROWS.map((row, r) => (
      <div key={r} dir="rtl" style={{display: 'flex', justifyContent: 'center', gap: 8, marginBottom: 12}}>
        {row.split('').map((ch, i) => {
          const idx = r * 11 + i;
          const on = idx === activeKey;
          return (
            <div
              key={i}
              style={{
                width: 48,
                height: 70,
                borderRadius: 10,
                background: on ? C.teal : C.white,
                boxShadow: '0 3px 0 #9DA3AD',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                fontFamily: HEEBO,
                fontWeight: 500,
                fontSize: 32,
                color: C.bg,
                transform: on ? 'translateY(-6px) scale(1.15)' : undefined,
              }}
            >
              {ch}
            </div>
          );
        })}
      </div>
    ))}
    <div style={{margin: '0 auto', width: 330, height: 66, borderRadius: 10, background: C.white, boxShadow: '0 3px 0 #9DA3AD'}} />
  </div>
);

/** the app screen, reused as the first frames of scene 4 */
export const MouzaAppScreen: React.FC<{frame: number}> = ({frame}) => {
  const {fps} = useVideoConfig();
  const chars = Math.max(0, Math.min(TYPED.length, Math.floor(((frame - TYPE_START) / (TYPE_END - TYPE_START)) * TYPED.length)));
  const typed = TYPED.slice(0, chars);
  const cursorOn = Math.floor(frame / 8) % 2 === 0;
  const press = spring({frame: frame - BUTTON_TAP, fps, config: POP});
  const btnScale = frame >= BUTTON_TAP && frame < BUTTON_TAP + 8 ? 0.9 + 0.1 * press : 1;
  const kbOut = spring({frame: frame - BUTTON_TAP, fps, config: BOUNCE});
  const activeKey = frame >= TYPE_START && frame < TYPE_END ? Math.floor(random(`key${Math.floor(frame / 3)}`) * 29) : -1;

  return (
    <AbsoluteFill style={{background: '#F6F4EF'}}>
      <div style={{position: 'absolute', top: 72, right: 34}}>
        <MouzaWordmark height={50} color={C.bg} />
      </div>
      <div dir="rtl" style={{position: 'absolute', top: 140, right: 34, left: 34, textAlign: 'right', fontFamily: HEEBO, fontWeight: 700, fontSize: 44, color: C.bg}}>
        על מה הסרטון?
      </div>
      {/* input */}
      <div
        dir="rtl"
        style={{
          position: 'absolute',
          top: 208,
          right: 30,
          left: 30,
          minHeight: 110,
          padding: '18px 22px',
          background: C.white,
          borderRadius: 24,
          border: `4px solid ${C.teal}`,
          textAlign: 'right',
          fontFamily: HEEBO,
          fontWeight: 500,
          fontSize: 38,
          lineHeight: 1.3,
          color: C.bg,
        }}
      >
        {typed}
        <span style={{display: 'inline-block', width: 4, height: 42, background: C.teal, marginRight: 4, verticalAlign: 'middle', opacity: cursorOn && frame < BUTTON_TAP + 4 ? 1 : 0}} />
      </div>
      {/* generate button */}
      <div
        dir="rtl"
        style={{
          position: 'absolute',
          top: 352,
          right: 30,
          left: 30,
          height: 78,
          borderRadius: 39,
          background: frame >= BUTTON_TAP ? C.yellow : C.teal,
          color: C.bg,
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontFamily: HEEBO,
          fontWeight: 800,
          fontSize: 38,
          transform: `scale(${btnScale})`,
          border: `4px solid ${C.bg}`,
          boxShadow: `0 6px 0 ${C.bg}`,
        }}
      >
        כתוב לי תסריט ✨
      </div>
      {/* script, line by line */}
      {frame >= LINE_AT[0] - 2 ? (
        <div dir="rtl" style={{position: 'absolute', top: 458, right: 30, left: 30, textAlign: 'right', fontFamily: HEEBO, fontWeight: 700, fontSize: 30, color: C.gray}}>
          התסריט שלך:
        </div>
      ) : null}
      <div style={{position: 'absolute', top: 504, right: 30, left: 30, display: 'flex', flexDirection: 'column', gap: 14}}>
        {SCRIPT_LINES.map((line, i) => {
          if (frame < LINE_AT[i]) return null;
          const s = spring({frame: frame - LINE_AT[i], fps, config: POP});
          return (
            <div
              key={i}
              dir="rtl"
              style={{
                background: C.white,
                borderRadius: 18,
                borderRight: `10px solid ${[C.teal, C.yellow, C.red][i]}`,
                padding: '12px 18px',
                fontFamily: HEEBO,
                fontWeight: 500,
                fontSize: 34,
                lineHeight: 1.25,
                color: C.bg,
                textAlign: 'right',
                transform: `translateX(${(1 - s) * 160}px) scale(${0.7 + 0.3 * s})`,
                transformOrigin: '100% 50%',
                boxShadow: '0 4px 0 rgba(0,0,0,0.12)',
              }}
            >
              {line}
            </div>
          );
        })}
      </div>
      <Keyboard y={820 + kbOut * 480} activeKey={activeKey} />
    </AbsoluteFill>
  );
};

/** Yaniv's finger: hovers over the keyboard, taps, then hits the button */
const Finger: React.FC<{x: number; y: number; tap: number}> = ({x, y, tap}) => (
  <svg width={260} height={700} viewBox="0 0 260 700" style={{position: 'absolute', left: x - 60, top: y - 20 + tap * 18, overflow: 'visible'}}>
    <path d="M 120 700 L 140 330 L 250 330 L 260 700 Z" fill={C.shirt} stroke={C.ink} strokeWidth={8} />
    <path d="M 40 60 Q 60 0 100 30 L 150 290 L 210 260 Q 250 300 240 360 L 150 380 Q 100 360 90 300 Z" fill={C.skin} stroke={C.ink} strokeWidth={8} strokeLinejoin="round" />
    <ellipse cx={66} cy={44} rx={20} ry={16} fill="#F7D6C2" transform="rotate(-20 66 44)" />
    <path d="M 160 300 Q 190 290 200 270" stroke={C.skinShade} strokeWidth={6} fill="none" />
  </svg>
);

export const Scene3Solution: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const enter = spring({frame: frame + 2, fps, config: BOUNCE});

  // finger path: keyboard taps while typing, then the button
  const typing = frame >= TYPE_START && frame < TYPE_END;
  const k = Math.floor(frame / 3);
  const fx = typing ? 330 + random(`fx${k}`) * 330 : frame < BUTTON_TAP + 10 ? 560 : 720;
  const fy = typing ? 1290 + random(`fy${k}`) * 170 : frame < BUTTON_TAP + 10 ? 770 : 1500;
  const tap = typing ? (frame % 3 === 0 ? 1 : 0) : frame >= BUTTON_TAP && frame < BUTTON_TAP + 4 ? 1 : 0;

  return (
    <AbsoluteFill>
      <HandheldShake amp={3}>
        <div style={{position: 'absolute', inset: 0, transform: 'scale(1.25)'}}>
          <Room blur={10} dim={0.25} />
        </div>
        {/* over-the-shoulder: back of Yaniv's head + shoulder, out of focus */}
        <svg viewBox="0 0 1080 1920" width={1080} height={1920} style={{position: 'absolute', inset: 0, filter: 'blur(6px)'}}>
          <path d="M -100 1920 L -60 1420 Q 0 1300 200 1290 L 300 1920 Z" fill={C.shirt} />
          <ellipse cx={60} cy={1080} rx={230} ry={270} fill={C.hair} />
          <ellipse cx={240} cy={1100} rx={40} ry={70} fill={C.skinShade} />
        </svg>
        <div style={{position: 'absolute', inset: 0, transform: `translateY(${(1 - enter) * 700}px) rotate(${(1 - enter) * 8}deg)`}}>
          <PhoneMockup x={PHONE.x} y={PHONE.y} width={PHONE.w} height={PHONE.h} rotate={-2}>
            <MouzaAppScreen frame={frame} />
          </PhoneMockup>
          {/* holding hand on the left edge */}
          <svg viewBox="0 0 1080 1920" width={1080} height={1920} style={{position: 'absolute', inset: 0, overflow: 'visible'}}>
            <path d="M 120 1920 Q 150 1640 236 1520" stroke={C.ink} strokeWidth={130} fill="none" strokeLinecap="round" />
            <path d="M 120 1920 Q 150 1640 236 1520" stroke={C.skin} strokeWidth={112} fill="none" strokeLinecap="round" />
            {[1240, 1320, 1400].map((y) => (
              <ellipse key={y} cx={238} cy={y} rx={34} ry={30} fill={C.skin} stroke={C.ink} strokeWidth={7} />
            ))}
          </svg>
          <Finger x={fx} y={fy} tap={tap} />
        </div>
      </HandheldShake>
      <Burst start={BUTTON_TAP} cx={560} cy={PHONE.y + 22 + 390} r={70} />
      <KineticText text="כותבים משפט אחד." cues={[6, 14, 22]} appearAt={2} top={SAFE.top + 4} fontSize={98} tilt={-1.5} holdLast={40} />
    </AbsoluteFill>
  );
};
