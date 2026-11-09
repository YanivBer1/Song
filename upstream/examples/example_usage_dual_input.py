"""
Example: Using the AI-SL API with both text and file inputs

This demonstrates how the Gradio interface can handle both text input
and file uploads, using whichever one is provided.
"""

from gradio_client import Client
import requests


def test_text_input():
    """
    Example 1: Using text input
    """
    print("=== Testing Text Input ===")
    
    # Connect to your hosted app
    client = Client("https://huggingface.co/spaces/your-username/your-space")
    
    # Test with text input
    text_input = "Hello world! This is a test of the text input functionality."
    
    # Call the interface with text input
    result = client.predict(
        text_input,  # Text input
        None,        # File input (None)
        True,        # Use R2 storage
        api_name="/predict"
    )
    
    # Process results
    json_data, video_url, download_html = result
    print(f"Status: {json_data['status']}")
    print(f"Video URL: {video_url}")
    
    return video_url


def test_file_input():
    """
    Example 2: Using file input
    """
    print("=== Testing File Input ===")
    
    # Connect to your hosted app
    client = Client("https://huggingface.co/spaces/your-username/your-space")
    
    # Test with file input
    file_path = "example_document.txt"
    
    # Call the interface with file input
    result = client.predict(
        "",          # Text input (empty)
        file_path,   # File input
        True,        # Use R2 storage
        api_name="/predict"
    )
    
    # Process results
    json_data, video_url, download_html = result
    print(f"Status: {json_data['status']}")
    print(f"Video URL: {video_url}")
    
    return video_url


def test_priority_logic():
    """
    Example 3: Testing the priority logic
    """
    print("=== Testing Priority Logic ===")
    
    # Connect to your hosted app
    client = Client("https://huggingface.co/spaces/your-username/your-space")
    
    # Test with both inputs (text should take priority)
    text_input = "This text should be processed instead of the file."
    file_path = "example_document.txt"
    
    # Call the interface with both inputs
    result = client.predict(
        text_input,  # Text input
        file_path,   # File input
        True,        # Use R2 storage
        api_name="/predict"
    )
    
    # Process results
    json_data, video_url, download_html = result
    print(f"Status: {json_data['status']}")
    print(f"Gloss: {json_data['gloss']}")
    print(f"Video URL: {video_url}")
    
    return video_url


def download_video(video_url, output_path):
    """
    Download a video from URL
    """
    try:
        response = requests.get(video_url, stream=True)
        response.raise_for_status()
        
        with open(output_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Video downloaded to: {output_path}")
        return True
    except Exception as e:
        print(f"Error downloading video: {e}")
        return False


def main():
    """
    Run all examples
    """
    print("AI-SL Dual Input Testing")
    print("=" * 50)
    
    # Test text input
    text_video_url = test_text_input()
    if text_video_url:
        download_video(text_video_url, "text_input_video.mp4")
    
    print("\n" + "-" * 50 + "\n")
    
    # Test file input
    file_video_url = test_file_input()
    if file_video_url:
        download_video(file_video_url, "file_input_video.mp4")
    
    print("\n" + "-" * 50 + "\n")
    
    # Test priority logic
    priority_video_url = test_priority_logic()
    if priority_video_url:
        download_video(priority_video_url, "priority_test_video.mp4")
    
    print("\n" + "=" * 50)
    print("Testing complete!")


if __name__ == "__main__":
    main() 