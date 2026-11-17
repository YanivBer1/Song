import React from 'react';
import {AbsoluteFill, spring, useCurrentFrame, useVideoConfig} from 'remotion';
import {CharacterYaniv} from './CharacterYaniv';
import {DoodleCircle} from './Doodles';
import {HandheldShake} from './HandheldShake';
import {MouzaMark} from './MouzaLogo';
import {Room} from './Room';
import {C, HEEBO, POP, W} from '../theme';

// COVER – export the still at frame 24 (0.8 s):
//   npx remotion still MouzaCover out/cover.png --frame=24
// Yaniv mid-laugh / mid-fail, phone in hand, blurry room, doodle circle
// round his face, "אני מצויר." big in the top third, tiny logo bottom-left.
// Built to survive the 3:4 profile-grid crop (y 240–1680).

export const CoverFrame: React.FC = () => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const textIn = spring({frame: frame - 4, fps, config: POP});
  const yIn = spring({frame, fps, config: POP});

  return (
    <AbsoluteFill style={{background: C.bg}}>
      <HandheldShake amp={2} zoom={1.08}>
        <AbsoluteFill style={{transform: 'scale(1.1)'}}>
          <Room blur={9} />
        </AbsoluteFill>
        <div style={{position: 'absolute', left: -60, top: 300, transform: `rotate(${4 * yIn}deg)`}}>
          <CharacterYaniv width={1240} expression="laugh" phone={1} tilt={-6} squash={0.96 + 0.04 * yIn} />
        </div>
      </HandheldShake>

      {/* doodle circle around the face */}
      <DoodleCircle start={6} cx={598} cy={980} rx={360} ry={400} color={C.yellow} width={16} />

      {/* "אני מצויר." – top third, white with black outline */}
      <div
        dir="rtl"
        style={{
          position: 'absolute',
          top: 300,
          left: 0,
          width: W,
          textAlign: 'center',
          fontFamily: HEEBO,
          fontWeight: 800,
          fontSize: 190,
          lineHeight: 1,
          color: C.white,
          WebkitTextStroke: `26px #000`,
          paintOrder: 'stroke fill',
          textShadow: '0 14px 0 rgba(0,0,0,0.6)',
          transform: `rotate(-4deg) scale(${0.5 + 0.5 * textIn})`,
        }}
      >
        אני מצויר.
      </div>

      {/* small logo, bottom-left, 8% width */}
      <div style={{position: 'absolute', left: 64, top: 1560}}>
        <MouzaMark size={Math.round(W * 0.08)} />
      </div>
    </AbsoluteFill>
  );
};
