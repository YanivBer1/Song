#!/usr/bin/env python3
"""
Test script to verify H.264 video encoding for browser compatibility
"""

import cv2
import numpy as np
import tempfile
import os
from video_gen import get_video_writer
from app import verify_video_format


def test_h264_encoding():
    """Test H.264 video encoding and verify browser compatibility"""
    
    print("Testing H.264 video encoding for browser compatibility...")
    
    # Create a simple test video
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_path = temp_file.name
    temp_file.close()
    
    try:
        # Create a simple test video with colored rectangles
        out = get_video_writer(temp_path, fps=30.0, width=640, height=480)
        
        if not out.isOpened():
            print("ERROR: Could not create video writer")
            return False
        
        # Create 3 seconds of video (90 frames at 30 fps)
        for frame_num in range(90):
            # Create a frame with changing colors
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            
            # Create a moving colored rectangle
            color = (
                int(255 * (frame_num % 30) / 30),  # Red
                int(255 * ((frame_num + 10) % 30) / 30),  # Green
                int(255 * ((frame_num + 20) % 30) / 30)   # Blue
            )
            
            # Draw a rectangle that moves across the screen
            x = int((frame_num % 60) * 640 / 60)
            cv2.rectangle(frame, (x, 200), (x + 100, 300), color, -1)
            
            # Add text
            cv2.putText(frame, f"Frame {frame_num}", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            out.write(frame)
        
        out.release()
        
        # Verify the video format
        is_compatible, message = verify_video_format(temp_path)
        print(f"Video format verification: {message}")
        
        if is_compatible:
            print("✅ SUCCESS: Video is H.264 encoded and browser compatible!")
            
            # Get video info
            cap = cv2.VideoCapture(temp_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            
            print(f"Video properties:")
            print(f"  - Resolution: {width}x{height}")
            print(f"  - FPS: {fps}")
            print(f"  - Frame count: {frame_count}")
            print(f"  - Duration: {frame_count/fps:.2f} seconds")
            print(f"  - File size: {os.path.getsize(temp_path)} bytes")
            
            return True
        else:
            print("❌ FAILED: Video is not browser compatible")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return False
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)
            print(f"Cleaned up test file: {temp_path}")


def test_codec_availability():
    """Test which video codecs are available"""
    
    print("\nTesting available video codecs...")
    
    codecs_to_test = [
        ('avc1', 'H.264 (best for browsers)'),
        ('mp4v', 'MPEG-4'),
        ('XVID', 'XVID'),
        ('MJPG', 'Motion JPEG'),
        ('H264', 'H.264 alternative')
    ]
    
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_path = temp_file.name
    temp_file.close()
    
    available_codecs = []
    
    for codec_name, description in codecs_to_test:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec_name)
            out = cv2.VideoWriter(temp_path, fourcc, 30.0, (640, 480))
            
            if out.isOpened():
                available_codecs.append((codec_name, description))
                print(f"✅ {codec_name}: {description}")
                out.release()
            else:
                print(f"❌ {codec_name}: {description} (not working)")
                out.release()
                
        except Exception as e:
            print(f"❌ {codec_name}: {description} (error: {e})")
    
    # Clean up
    if os.path.exists(temp_path):
        os.unlink(temp_path)
    
    print(f"\nAvailable codecs: {len(available_codecs)}")
    return available_codecs


if __name__ == "__main__":
    print("=" * 60)
    print("H.264 Video Encoding Test for Browser Compatibility")
    print("=" * 60)
    
    # Test available codecs
    available_codecs = test_codec_availability()
    
    # Test H.264 encoding
    success = test_h264_encoding()
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed! Videos should work in Chrome and Firefox.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    print("=" * 60) 