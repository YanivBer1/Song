import React from 'react';
import {Composition} from 'remotion';
import {CoverFrame} from './components/CoverFrame';
import {MouzaVideo, MouzaVideoProps} from './MouzaVideo';
import {DURATION, FPS, H, W} from './theme';

// Main: Mouza11Attempts. A/B Trial-Reels variants: MouzaHookA…E (same body,
// different 0–2.8 s hook). Cover: MouzaCover (export still at frame 24).

const VARIANTS: {id: string; hookId: MouzaVideoProps['hookId']}[] = [
  {id: 'Mouza11Attempts', hookId: 'main'},
  {id: 'MouzaHookA', hookId: 'A'},
  {id: 'MouzaHookB', hookId: 'B'},
  {id: 'MouzaHookC', hookId: 'C'},
  {id: 'MouzaHookD', hookId: 'D'},
  {id: 'MouzaHookE', hookId: 'E'},
];

export const RemotionRoot: React.FC = () => (
  <>
    {VARIANTS.map((v) => (
      <Composition
        key={v.id}
        id={v.id}
        component={MouzaVideo}
        durationInFrames={DURATION}
        fps={FPS}
        width={W}
        height={H}
        defaultProps={{hookId: v.hookId}}
      />
    ))}
    <Composition id="MouzaCover" component={CoverFrame} durationInFrames={30} fps={FPS} width={W} height={H} />
  </>
);
