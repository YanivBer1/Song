import React from 'react';
import {C} from '../theme';
import {SketchRect} from './Doodles';

// Phone drawn in code (no screenshots). Children render inside the screen,
// in screen-local pixels (0,0 = top-left of the screen glass).

export const PHONE_BEZEL = 22;

export const PhoneMockup: React.FC<{
  x: number;
  y: number;
  width: number;
  height: number;
  rotate?: number;
  scale?: number;
  screenBg?: string;
  children?: React.ReactNode;
  sketch?: boolean;
}> = ({x, y, width, height, rotate = 0, scale = 1, screenBg = '#F6F4EF', children, sketch = true}) => (
  <div
    style={{
      position: 'absolute',
      left: x,
      top: y,
      width,
      height,
      transform: `rotate(${rotate}deg) scale(${scale})`,
      transformOrigin: '50% 60%',
    }}
  >
    {/* body */}
    <div
      style={{
        position: 'absolute',
        inset: 0,
        background: '#17181D',
        borderRadius: 78,
        border: `6px solid ${C.ink}`,
        boxShadow: `inset 0 0 0 4px #3A3F4B, 18px 24px 0 rgba(0,0,0,0.45)`,
      }}
    />
    {/* side buttons */}
    <div style={{position: 'absolute', left: -12, top: 220, width: 10, height: 90, background: '#2A2D35', borderRadius: 4}} />
    <div style={{position: 'absolute', right: -12, top: 260, width: 10, height: 140, background: '#2A2D35', borderRadius: 4}} />
    {/* screen */}
    <div
      style={{
        position: 'absolute',
        left: PHONE_BEZEL,
        top: PHONE_BEZEL,
        width: width - PHONE_BEZEL * 2,
        height: height - PHONE_BEZEL * 2,
        background: screenBg,
        borderRadius: 58,
        overflow: 'hidden',
      }}
    >
      {children}
      {/* notch pill */}
      <div
        style={{
          position: 'absolute',
          top: 18,
          left: '50%',
          width: 150,
          height: 40,
          transform: 'translateX(-50%)',
          background: '#0B0B0E',
          borderRadius: 20,
        }}
      />
    </div>
    {sketch ? <SketchRect x={-8} y={-8} w={width + 16} h={height + 16} color="#FFFFFF" width={4} seed="phone" /> : null}
  </div>
);
