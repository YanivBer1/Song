"""
Example: How to programmatically access video output from the AI-SL API

This file demonstrates different ways users can receive and process video 
output from the Gradio interface.
"""

import requests
import base64
from pathlib import Path


def download_video_from_url(video_url, output_path="downloaded_video.mp4"):
    """
    Download a video from a public URL
    """
    try:
        response = requests.get(video_url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Video downloaded to: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None


def save_base64_video(base64_data, output_path="video_from_base64.mp4"):
    """
    Save a base64-encoded video to a file
    """
    try:
        # Remove the data URL prefix if present
        if base64_data.startswith('data:video/mp4;base64,'):
            base64_data = base64_data.split(',')[1]
        
        # Decode and save
        video_data = base64.b64decode(base64_data)
        with open(output_path, 'wb') as f:
            f.write(video_data)
        
        print(f"Video saved from base64 to: {output_path}")
        return output_path
    except Exception as e:
        print(f"Error saving base64 video: {e}")
        return None


def process_gradio_output(gradio_result):
    """
    Process the output from the Gradio interface
    
    gradio_result should be a tuple: (json_data, video_output, download_html)
    """
    json_data, video_output, download_html = gradio_result
    
    print("Processing Results:")
    print(f"Status: {json_data['status']}")
    print(f"Video Count: {json_data['video_count']}")
    print(f"Gloss: {json_data['gloss']}")
    
    # Handle video output based on format
    if json_data.get('video_format') == 'base64':
        # Video is base64 encoded
        print("Video format: Base64")
        video_path = save_base64_video(video_output, "asl_output.mp4")
    else:
        # Video is a URL (from R2 upload)
        print("Video format: URL")
        video_path = download_video_from_url(video_output, "asl_output.mp4")
    
    return video_path


# Example usage scenarios:

def example_1_direct_download():
    """
    Example 1: Direct download from R2 URL
    """
    print("=== Example 1: Direct Download ===")
    
    # Simulate getting a video URL from the interface
    video_url = "https://your-r2-endpoint.com/bucket/video.mp4"
    
    # Download the video
    video_path = download_video_from_url(video_url)
    
    if video_path:
        print(f"Video ready for processing: {video_path}")
        # Now you can use the video file for further processing
        # e.g., upload to another service, analyze with OpenCV, etc.


def example_2_base64_processing():
    """
    Example 2: Processing base64 video data
    """
    print("=== Example 2: Base64 Processing ===")
    
    # Simulate getting base64 data from the interface
    base64_video = ("data:video/mp4;base64,AAAAIGZ0eXBpc29tAAACAGlzb21pc28yYXZjMW1wNDEAAAAIZnJlZQAA...")  # noqa: E501
    
    # Save the video
    video_path = save_base64_video(base64_video)
    
    if video_path:
        print(f"Video ready for processing: {video_path}")


def example_3_programmatic_interface():
    """
    Example 3: Using the Gradio interface programmatically
    """
    print("=== Example 3: Programmatic Interface ===")
    
    # If you want to call the Gradio interface programmatically
    # You can use the gradio_client library
    
    try:
        from gradio_client import Client
        
        # Connect to your running Gradio interface
        client = Client("http://localhost:7860")  # Adjust URL as needed
        
        # Upload a document and get results
        result = client.predict(
            "path/to/your/document.pdf",  # File path
            api_name="/predict"  # Adjust based on your interface
        )
        
        # Process the results
        video_path = process_gradio_output(result)
        print(f"Processed video: {video_path}")
        
    except ImportError:
        print("gradio_client not installed. Install with: "
              "pip install gradio_client")
    except Exception as e:
        print(f"Error calling Gradio interface: {e}")


def example_4_video_processing():
    """
    Example 4: Further video processing
    """
    print("=== Example 4: Video Processing ===")
    
    # Once you have the video file, you can process it further
    video_path = "asl_output.mp4"
    
    if Path(video_path).exists():
        print(f"Processing video: {video_path}")
        
        # Example: Get video information
        try:
            import cv2
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            cap.release()
            
            print(f"Video info: {duration:.2f} seconds, {fps} FPS, "
                  f"{frame_count} frames")
            
        except ImportError:
            print("OpenCV not installed. Install with: "
                  "pip install opencv-python")
        
        # Example: Upload to another service
        # upload_to_youtube(video_path)
        # upload_to_drive(video_path)
        # etc.


if __name__ == "__main__":
    # Run examples
    example_1_direct_download()
    example_2_base64_processing()
    example_3_programmatic_interface()
    example_4_video_processing() 