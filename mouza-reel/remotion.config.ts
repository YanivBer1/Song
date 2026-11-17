import {Config} from '@remotion/cli/config';

// Export spec: H.264 MP4, 1080x1920, 30 fps, ~10 Mbps video, AAC 128 kbps.
// (Size/fps live in src/Root.tsx. Remotion writes 48 kHz audio by default.)
Config.setCodec('h264');
Config.setVideoImageFormat('jpeg');
Config.setJpegQuality(92);
Config.setVideoBitrate('10M');
Config.setAudioBitrate('128k');
Config.setPixelFormat('yuv420p');
Config.setOverwriteOutput(true);
