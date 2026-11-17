import React, {createContext, useContext} from 'react';
import {Html5Audio, random, Sequence, staticFile, useCurrentFrame, useVideoConfig} from 'remotion';
import {getStaticFiles} from '@remotion/studio';
import {useAudioData, visualizeAudio} from '@remotion/media-utils';
import type {VoLine} from './timing';

// ============================================================================
//  REAL-VOICE SYNC
//  If public/vo/<file> exists, it is played AND Yaniv's mouth is driven by
//  the loudness of YOUR recording, frame by frame (stutters, laughs and
//  pauses included). If the file is missing, the mouth flaps on the
//  scripted talk windows from timing.ts so you can preview before recording.
//  Mouth updates on twos (every 2 frames) – classic lo-fi animation timing.
// ============================================================================

const MouthCtx = createContext(0);
export const useMouth = () => useContext(MouthCtx);

export const hasStatic = (path: string) => getStaticFiles().some((f) => f.name === path);

const AudioMouth: React.FC<{vo: VoLine; children: React.ReactNode}> = ({vo, children}) => {
  const frame = useCurrentFrame();
  const {fps} = useVideoConfig();
  const src = staticFile(`vo/${vo.file}`);
  const audio = useAudioData(src);
  const local = Math.floor((frame - vo.offset) / 2) * 2;
  let level = 0;
  if (audio && local >= 0 && local / fps < audio.durationInSeconds - 0.05) {
    const bins = visualizeAudio({fps, frame: local, audioData: audio, numberOfSamples: 32, optimizeFor: 'speed'});
    // speech energy lives in the low/mid bins
    const speech = bins.slice(1, 12).reduce((a, b) => a + b, 0) / 11;
    level = Math.min(1, Math.max(0, (speech - 0.012) * 14));
  }
  return (
    <MouthCtx.Provider value={level}>
      <Sequence from={vo.offset} layout="none" name={`VO ${vo.file}`}>
        <Html5Audio src={src} />
      </Sequence>
      {children}
    </MouthCtx.Provider>
  );
};

const ScriptedMouth: React.FC<{vo: VoLine; children: React.ReactNode}> = ({vo, children}) => {
  const frame = useCurrentFrame();
  const q = Math.floor(frame / 2) * 2;
  let level = 0;
  for (const [a, b] of vo.talk) {
    if (q >= a && q <= b) {
      const edge = Math.min(1, (q - a) / 4, (b - q) / 4);
      level = (0.2 + 0.8 * random(`mouth-${vo.file}-${q}`)) * edge;
    }
  }
  return <MouthCtx.Provider value={level}>{children}</MouthCtx.Provider>;
};

/** Wrap a scene. Plays the VO line (if recorded) and provides mouth level to children. */
export const Voice: React.FC<{vo: VoLine; children: React.ReactNode}> = ({vo, children}) =>
  hasStatic(`vo/${vo.file}`) ? <AudioMouth vo={vo}>{children}</AudioMouth> : <ScriptedMouth vo={vo}>{children}</ScriptedMouth>;
