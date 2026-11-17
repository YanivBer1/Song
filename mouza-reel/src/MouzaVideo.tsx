import React from 'react';
import {AbsoluteFill, Sequence} from 'remotion';
import {HOOKS} from './hooks';
import {Scene1Hook} from './scenes/Scene1Hook';
import {Scene2Pain} from './scenes/Scene2Pain';
import {Scene3Solution} from './scenes/Scene3Solution';
import {Scene4Retake} from './scenes/Scene4Retake';
import {Scene5Proof} from './scenes/Scene5Proof';
import {Scene6Payoff} from './scenes/Scene6Payoff';
import {Scene7Loop} from './scenes/Scene7Loop';
import {Grain} from './scenes/shared';
import {Soundtrack} from './Soundtrack';
import {C} from './theme';
import {SCENES, VO} from './timing';
import {Voice} from './voice';

// ============================================================================
//  MOUZA – "11 ATTEMPTS" – 27 s / 810 frames / 1080×1920 / 30 fps
//  Scene 1 (the hook) is swapped per A/B variant. Frames 84–810 are the
//  shared body and identical in every composition.
// ============================================================================

export type MouzaVideoProps = {hookId: keyof typeof HOOKS};

export const MouzaVideo: React.FC<MouzaVideoProps> = ({hookId}) => {
  const hook = HOOKS[hookId];
  return (
    <AbsoluteFill style={{background: C.bg}}>
      <Sequence from={SCENES.hook.from} durationInFrames={SCENES.hook.dur} name={`1 HOOK (${hook.id})`}>
        <Voice vo={hook.vo}>
          <Scene1Hook hook={hook} />
        </Voice>
      </Sequence>

      <Sequence from={SCENES.pain.from} durationInFrames={SCENES.pain.dur} name="2 PAIN">
        <Voice vo={VO.pain}>
          <Scene2Pain />
        </Voice>
      </Sequence>

      <Sequence from={SCENES.solution.from} durationInFrames={SCENES.solution.dur} name="3 SOLUTION">
        <Voice vo={VO.solution}>
          <Scene3Solution />
        </Voice>
      </Sequence>

      <Sequence from={SCENES.retake.from} durationInFrames={SCENES.retake.dur} name="4 RETAKE">
        <Voice vo={VO.retake}>
          <Scene4Retake />
        </Voice>
      </Sequence>

      <Sequence from={SCENES.proof.from} durationInFrames={SCENES.proof.dur} name="5 BEFORE/AFTER">
        <Voice vo={VO.proof}>
          <Scene5Proof />
        </Voice>
      </Sequence>

      <Sequence from={SCENES.payoff.from} durationInFrames={SCENES.payoff.dur} name="6 PAYOFF">
        <Voice vo={VO.payoff}>
          <Scene6Payoff />
        </Voice>
      </Sequence>

      <Sequence from={SCENES.loop.from} durationInFrames={SCENES.loop.dur} name="7 LOOP + SEND">
        <Voice vo={VO.loop}>
          <Scene7Loop />
        </Voice>
      </Sequence>

      <Grain />
      <Soundtrack hook={hook} />
    </AbsoluteFill>
  );
};
