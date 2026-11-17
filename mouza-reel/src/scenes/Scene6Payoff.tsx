import React from 'react';
import {AbsoluteFill, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {CharacterYaniv, Expression} from '../components/CharacterYaniv';
import {KineticText} from '../components/KineticText';
import {Room} from '../components/Room';
import {BOUNCE, C, SAFE} from '../theme';
import {FinishedReel, usePop} from './shared';
import {REEL_BROLL, REEL_CUES, reelMouth} from './Scene5Proof';

// SCENE 6 – PAYOFF – frames 540–690 (18.0–23.0 s), local 0–150
// The "after" half explodes to full screen: the finished Mouza video plays
// exactly as it would on Reels. PiP: Yaniv in his dark room watching it
// on his own phone – smiling, then surprised it is actually good.

const PIP = {x: 60, y: 836, w: 236, h: 296};

const PipYaniv: React.FC = () => {
  const frame = useCurrentFrame();
  const s = usePop(12);
  let expr: Expression = 'smile';
  if (frame >= 36 && frame < 70) expr = 'surprised';
  else if (frame >= 70) expr = 'proud';
  if (frame < 12) return null;
  return (
    <div
      style={{
        position: 'absolute',
        left: PIP.x,
        top: PIP.y,
        width: PIP.w,
        height: PIP.h,
        borderRadius: 34,
        overflow: 'hidden',
        border: `7px solid ${C.white}`,
        boxShadow: `10px 12px 0 ${C.bg}`,
        transform: `scale(${s}) rotate(${-3 * s}deg)`,
        transformOrigin: '0% 100%',
        background: C.bg,
      }}
    >
      <div style={{position: 'absolute', left: -420, top: -600, width: 1080, height: 1920, transform: 'scale(0.36)'}}>
        <Room blur={3} dim={0.35} />
      </div>
      <div style={{position: 'absolute', left: -13, top: -24}}>
        <CharacterYaniv width={262} expression={expr} phone={1} look={[0, 0.8]} />
      </div>
    </div>
  );
};

export const Scene6Payoff: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const grow = spring({frame: frame + 1, fps, config: BOUNCE});
  const t = 150 + frame; // continuous with scene 5's right panel

  return (
    <AbsoluteFill style={{background: C.bg}}>
      <AbsoluteFill style={{clipPath: `inset(0 0 0 ${Math.max(0, 540 * (1 - grow))}px)`}}>
        <AbsoluteFill style={{transform: `translateX(${270 * (1 - grow)}px)`}}>
          <FinishedReel
            t={t}
            cues={REEL_CUES}
            broll={REEL_BROLL}
            captionTop={1144}
            captionLeft={SAFE.left + 20}
            captionWidth={910}
            captionSize={72}
            mouth={reelMouth(t)}
            yaniv={{left: 10, top: 330, width: 1060}}
          />
        </AbsoluteFill>
      </AbsoluteFill>
      <PipYaniv />
      <KineticText
        text="הסרטון הזה. בלי עורך. בלי צלם."
        cues={[8, 14, 34, 40, 62, 68]}
        appearAt={4}
        top={SAFE.top + 2}
        fontSize={88}
        tilt={-1.5}
        holdLast={40}
        circle={{word: 1, at: 18, color: C.yellow}}
      />
    </AbsoluteFill>
  );
};
