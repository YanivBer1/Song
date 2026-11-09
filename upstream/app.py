from document_to_gloss import DocumentToASLConverter
from document_parsing import DocumentParser
from vectorizer import Vectorizer
from video_gen import create_multi_stitched_video
import gradio as gr
import asyncio
import re
import boto3
import os
from botocore.config import Config
from dotenv import load_dotenv
import requests
import tempfile
import uuid
import base64

# Load environment variables from .env file
load_dotenv()

# Load R2/S3 environment secrets
R2_ASL_VIDEOS_URL = os.environ.get("R2_ASL_VIDEOS_URL")
R2_ENDPOINT = os.environ.get("R2_ENDPOINT")
R2_ACCESS_KEY_ID = os.environ.get("R2_ACCESS_KEY_ID")
R2_SECRET_ACCESS_KEY = os.environ.get("R2_SECRET_ACCESS_KEY")

# Validate that required environment variables are set
if not all([R2_ASL_VIDEOS_URL, R2_ENDPOINT, R2_ACCESS_KEY_ID, 
            R2_SECRET_ACCESS_KEY]):
    raise ValueError(
        "Missing required R2 environment variables. "
        "Please check your .env file."
    )

title = "AI-SL"
description = "Convert text to ASL!"
article = ("<p style='text-align: center'><a href='https://github.com/deenasun' "
           "target='_blank'>Deena Sun on Github</a></p>")
inputs = gr.File(label="Upload Document (pdf, txt, docx, or epub)")
outputs = [
    gr.JSON(label="Processing Results"), 
    gr.Video(label="ASL Video Output"),
    gr.HTML(label="Download Link")
]

parser = DocumentParser()
asl_converter = DocumentToASLConverter()
vectorizer = Vectorizer()
session = boto3.session.Session()

s3 = session.client(
    service_name='s3',
    region_name='auto',
    endpoint_url=R2_ENDPOINT,
    aws_access_key_id=R2_ACCESS_KEY_ID,
    aws_secret_access_key=R2_SECRET_ACCESS_KEY,
    config=Config(signature_version='s3v4')
)

def clean_gloss_token(token):
    """Clean a single gloss token"""
    if not token:
        return None
    
    # Remove punctuation and convert to lowercase
    cleaned = re.sub(r'[^\w\s]', '', token).lower().strip()
    
    # Remove extra whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    
    return cleaned if cleaned else None


def verify_video_format(video_path):
    """
    Verify that a video file is in a browser-compatible format (H.264 MP4)
    """
    try:
        import cv2
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return False, "Could not open video file"
        
        # Get video properties
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        cap.release()
        
        # Check if it's H.264
        if codec in ['avc1', 'H264', 'h264']:
            return True, f"Video is H.264 encoded ({codec})"
        else:
            return False, f"Video codec {codec} may not be browser compatible"
            
    except Exception as e:
        return False, f"Error checking video format: {e}"


def upload_video_to_r2(video_path, bucket_name="asl-videos"):
    """
    Upload a video file to R2 and return a public URL
    """
    try:
        # Verify video format for browser compatibility
        is_compatible, message = verify_video_format(video_path)
        print(f"Video format check: {message}")
        
        # Generate a unique filename
        file_extension = os.path.splitext(video_path)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        
        # Upload to R2
        with open(video_path, 'rb') as video_file:
            s3.upload_fileobj(
                video_file,
                bucket_name,
                unique_filename,
                ExtraArgs={
                    'ACL': 'public-read',
                    'ContentType': 'video/mp4; codecs="avc1.42E01E"',  # H.264
                    'CacheControl': 'max-age=86400',  # Cache for 24 hours
                    'ContentDisposition': 'inline'    # Force inline display
                })
        
        # Replace the endpoint with the domain for uploading
        if R2_ENDPOINT:
            public_domain = (R2_ENDPOINT.replace('https://', '')
                             .split('.')[0])
            video_url = (f"https://{public_domain}.r2.cloudflarestorage.com/"
                         f"{bucket_name}/{unique_filename}")

            print(f"Video uploaded to R2: {video_url}")
            public_video_url = f"{R2_ASL_VIDEOS_URL}/{unique_filename}"
            print(f"Public video url: {public_video_url}")

            return public_video_url
        else:
            print("R2_ENDPOINT is not configured")
            return None
        
    except Exception as e:
        print(f"Error uploading video to R2: {e}")
        return None

def video_to_base64(video_path):
    """
    Convert a video file to base64 string for direct download
    """
    try:
        with open(video_path, 'rb') as video_file:
            video_data = video_file.read()
            base64_data = base64.b64encode(video_data).decode('utf-8')
            return f"data:video/mp4;base64,{base64_data}"
    except Exception as e:
        print(f"Error converting video to base64: {e}")
        return None

def download_video_from_url(video_url):
    """
    Download a video from a public R2 URL
    Returns the local file path where the video is saved
    """
    try:
        # Create a temporary file with .mp4 extension
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        temp_path = temp_file.name
        temp_file.close()
        
        # Download the video
        print(f"Downloading video from: {video_url}")
        response = requests.get(video_url, stream=True)
        response.raise_for_status()
        
        # Save to temporary file
        with open(temp_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Video downloaded to: {temp_path}")
        return temp_path
        
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None


def cleanup_temp_video(file_path):
    """
    Clean up temporary video file
    """
    try:
        if file_path and os.path.exists(file_path):
            os.unlink(file_path)
            print(f"Cleaned up: {file_path}")
    except Exception as e:
        print(f"Error cleaning up file: {e}")


def determine_input_type(input_data):
    """
    Determine the type of input data and return a standardized format.
    Returns: (input_type, processed_data) where input_type is 'text', 
    'file_path', or 'file_object'
    """
    if isinstance(input_data, str):
        # Check if it's a file path (contains file extension)
        if any(ext in input_data.lower() for ext in ['.pdf', '.txt', '.docx', '.doc', '.epub']):
            return 'file_path', input_data
        # Check if it's a string representation of a gradio.FileData dict
        elif input_data.startswith('{') and 'gradio.FileData' in input_data:
            try:
                import ast
                import json
                # Try to parse as JSON first
                try:
                    file_data = json.loads(input_data)
                except json.JSONDecodeError:
                    # Fall back to ast.literal_eval for safer parsing
                    file_data = ast.literal_eval(input_data)
                
                if isinstance(file_data, dict) and 'path' in file_data:
                    print(f"Parsed FileData: {file_data}")
                    return 'file_path', file_data['path']
            except (ValueError, SyntaxError, json.JSONDecodeError) as e:
                print(f"Error parsing FileData string: {e}")
                print(f"Input data: {input_data}")
                pass
        else:
            return 'text', input_data.strip()
    elif isinstance(input_data, dict) and 'path' in input_data:
        # This is a gradio.FileData object from API calls
        return 'file_path', input_data['path']
    elif hasattr(input_data, 'name'):
        # This is a regular file object
        return 'file_path', input_data.name
    else:
        return 'unknown', None


def process_input(input_data):
    """
    Extract text content from various input types.
    Returns the text content ready for ASL conversion.
    """
    input_type, processed_data = determine_input_type(input_data)
    
    if input_type == 'text':
        return processed_data
    elif input_type == 'file_path':
        try:
            print(f"Processing file: {processed_data}")
            # Use document converter for all file types
            gloss = asl_converter.convert_document(processed_data)
            print(f"Converted gloss: {gloss[:100]}...")
            return gloss
        except Exception as e:
            print(f"Error processing file: {e}")
            return None
    else:
        print(f"Unsupported input type: {type(input_data)}")
        return None



async def parse_vectorize_and_search_unified(input_data):
    """
    Unified function that handles both text and file inputs
    """    
    # Process the input to get gloss
    gloss = process_input(input_data)
    if not gloss:
        return {
            "status": "error",
            "message": "Failed to process input"
        }, None
    
    print("ASL", gloss)
    
    # Split by spaces and clean each token
    gloss_tokens = gloss.split()
    cleaned_tokens = []
    
    for token in gloss_tokens:
        cleaned = clean_gloss_token(token)
        if cleaned:  # Only add non-empty tokens
            cleaned_tokens.append(cleaned)
    
    print("Cleaned tokens:", cleaned_tokens)

    videos = []
    video_files = []  # Store local file paths for stitching
    
    for g in cleaned_tokens:
        print(f"Processing {g}")
        try:
            result = await vectorizer.vector_query_from_supabase(query=g)
            print("result", result)
            if result.get("match", False):
                video_url = result["video_url"]
                videos.append(video_url)
                
                # Download the video
                local_path = download_video_from_url(video_url)
                if local_path:
                    video_files.append(local_path)
                    
        except Exception as e:
            print(f"Error processing {g}: {e}")
            continue
    
    # Create stitched video if we have multiple videos
    stitched_video_path = None
    if len(video_files) > 1:
        try:
            print(f"Creating stitched video from {len(video_files)} videos...")
            stitched_video_path = tempfile.NamedTemporaryFile(
                delete=False, suffix='.mp4'
            ).name
            create_multi_stitched_video(video_files, stitched_video_path)
            print(f"Stitched video created: {stitched_video_path}")
        except Exception as e:
            print(f"Error creating stitched video: {e}")
            stitched_video_path = None
    elif len(video_files) == 1:
        # If only one video, just use it directly
        stitched_video_path = video_files[0]
    
    # Upload final video to R2 and get public URL
    video_download_url = None
    if stitched_video_path:
        video_download_url = upload_video_to_r2(stitched_video_path)
        # Don't clean up the local file yet - let frontend use it first
    
    # Clean up individual video files after stitching
    for video_file in video_files:
        if video_file != stitched_video_path:  # Don't delete the final output
            cleanup_temp_video(video_file)
    
    video64 = video_to_base64(stitched_video_path)
    # Return simplified results
    return {
        "status": "success",
        "videos": videos,
        "video_count": len(videos),
        "gloss": gloss,
        "cleaned_tokens": cleaned_tokens,
        "video_download_url": video_download_url,
        "video_as_base_64": video64
    }, stitched_video_path


def parse_vectorize_and_search_unified_sync(input_data):
    return asyncio.run(parse_vectorize_and_search_unified(input_data))


def predict_unified(input_data):
    """
    Unified prediction function that handles both text and file inputs
    """
    try:
        if input_data is None:
            return {
                "status": "error",
                "message": "Please provide text or upload a document"
            }, None
        
        # Use the unified processing function
        result = parse_vectorize_and_search_unified_sync(input_data)
        
        # Get the results
        json_data, local_video_path = result
        
        # If we have a local video path, use it directly for Gradio
        if local_video_path and json_data.get("status") == "success":
            # Schedule cleanup of the video file after a delay
            # This gives Gradio time to load and display the video
            import threading
            import time
            
            def delayed_cleanup(video_path):
                time.sleep(30)  # Wait 30 seconds before cleanup
                cleanup_temp_video(video_path)
            
            # Start cleanup thread
            cleanup_thread = threading.Thread(
                target=delayed_cleanup, 
                args=(local_video_path,)
            )
            cleanup_thread.daemon = True
            cleanup_thread.start()
            
            return json_data, local_video_path
        
        return result
        
    except Exception as e:
        print(f"Error in predict_unified function: {e}")
        return {
            "status": "error",
            "message": f"An error occurred: {str(e)}"
        }, None


# Create the Gradio interface
def create_interface():
    """Create and configure the Gradio interface"""
    
    # Create the interface
    interface = gr.Interface(
        fn=predict,
        inputs=[
            gr.Textbox(
                label="Enter text to convert to ASL",
                placeholder="Type or paste your text here...",
                lines=5
            ),
            gr.File(
                label="Upload Document (pdf, txt, docx, or epub)",
                file_types=[".pdf", ".txt", ".docx", ".epub"]
            )
        ],
        outputs=[
            gr.JSON(label="Results"),
            gr.Video(label="ASL Video")
        ],
        title=title,
        description=description,
        article=article
    )
    
    return interface


# Add a predict function for Hugging Face API access
def predict(text, file):
    """
    Predict function for Hugging Face API access.
    This function will be available as the /predict endpoint.
    """
    # Determine which input to use
    if text and text.strip():
        # Use text input
        input_data = text.strip()
    elif file is not None:
        # Use file input - let the centralized processor handle the type
        input_data = file
    else:
        # No input provided
        return {
            "status": "error",
            "message": "Please provide either text or upload a file"
        }, None
    
    print("Input to the prediction function", input_data)
    print("Input type:", type(input))
    # Process using the unified function
    return predict_unified(input_data)


# For Hugging Face Spaces, use the Interface
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True  # Set to True for local testing with public URL
    )
