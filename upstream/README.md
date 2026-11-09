---
title: AI-powered ASL text-to-video Generator
emoji: 🐻
colorFrom: blue
colorTo: yellow
sdk: gradio
sdk_version: 5.34.2
app_file: app.py
pinned: false
license: apache-2.0
---

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference

# AI-SL API

Convert natural language English into American Sign Language (ASL) videos using AI!

View our full repo for the AI-SL Project created for the **Berkeley AI Hackathon 2025** 🚀 here: [AI-SL Repo](https://github.com/deenasun/ai-sl)

![Team photo from Berkeley AI Hackathon 2025](team_photo.jpeg)

## Features

### Dual Input Support with Optional File Upload
The app accepts both text input and file uploads with flexible options:

- **Text Input**: Type or paste text directly into the interface (always available)
- **File Upload**: Upload documents (PDF, TXT, DOCX, EPUB)

### Video Output Options

The Gradio interface provides multiple ways for users to receive and download the generated ASL videos:

#### 1. R2 Cloud Storage
- Videos are automatically uploaded to Cloudflare R2 storage
- Returns a public URL that users can download directly
- Videos persist and can be shared via URL
- Includes a styled download button in the interface

#### 2. Base64 Encoding (Alternative)
- Videos are embedded as base64 data directly in the response
- No external storage required
- Good for smaller videos or when you want to avoid cloud storage
- Can be downloaded directly from the interface

#### 3. Programmatic Access
Users can access the video output programmatically using:

```python
from gradio_client import Client

# Connect to the running interface
client = Client("http://localhost:7860")

# Upload a document and get results
result = client.predict(
    "path/to/document.pdf",
    api_name="/predict"
)

# The result contains: (json_data, video_output)
json_data, video_url = result

# Download the video
import requests
response = requests.get(video_url)
with open("asl_video.mp4", "wb") as f:
    f.write(response.content)
```

## Example Usage

### Web Interface
1. Visit your Space URL
2. Choose input method:
   - **Text**: Type or paste text in the text box (always available)
   - **File**: Check "Enable file upload" and upload a document (optional)
3. Click "Submit"
4. Download the resulting video

### Programmatic Access with Optional File Upload

```python
from gradio_client import Client

# Connect to your hosted app
from gradio_client import Client, handle_file
client = Client("deenasun/ai-sl-api")

# Text input only (file upload disabled)
result = client.predict(
    text="Hello world! This is a test.",  # Text input
    file=None,                            # File input (None since disabled)
    api_name="/predict"
)

# File input only (file upload enabled)
result = client.predict(
    text="",                              # Text input (empty)
    file=handle_file("document.pdf"),     # File input
    api_name="/predict"
)

# Both inputs (text takes priority)
result = client.predict(
    "Quick text",                    # Text input
    "document.pdf",                  # File input
    api_name="/predict"
)
```

See `example_usage.py` and `example_usage_dual_input.py` for complete examples of how to:
- Download videos from URLs
- Process base64 video data
- Use the interface programmatically
- Perform further video processing
- Handle both text and file inputs
- Use optional file upload functionality

## Requirements

- Python 3.7+
- Required packages listed in `requirements.txt`
- Cloudflare R2 credentials (for cloud storage option)
- Supabase credentials for video database

## Setup

1. Install dependencies: `pip install -r requirements.txt`
2. Set up environment variables in `.env` file
3. Run the interface: `python app.py`

## Video Processing

Once you have the video file, you can:
- Upload to YouTube, Google Drive, or other services
- Analyze with OpenCV for computer vision tasks
- Convert to different formats
- Extract frames for further processing
- Add subtitles or overlays
