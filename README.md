# Video Bubble Subtitler

This Gradio application automatically transcribes the audio from your uploaded video and generates word-by-word "bubble" subtitles, overlaid at a fixed position near the bottom of the video.

## Features

- **Automated Speech Recognition (ASR):** Utilizes Hugging Face's Whisper model for accurate transcription.
- **Dynamic Bubble Generation:** Creates visually appealing, semi-transparent speech bubbles for each word.
- **Fixed Positioning:** Bubbles appear consistently centered horizontally and 12% from the bottom edge vertically, ensuring a clean and predictable layout.
- **Random Bubble Colors:** Each bubble gets a random, vibrant color with contrasting text for better visibility.
- **User-Friendly Interface:** Built with Gradio for easy video uploads and processed video downloads.

## How it Works

1.  **Audio Extraction:** FFmpeg is used to extract the audio track from the uploaded video.
2.  **Transcription:** The audio is then processed by a pre-trained Whisper model (e.g., `openai/whisper-small`) to generate a transcript with precise word-level timestamps.
3.  **Bubble Image Creation:** For each transcribed word, a custom bubble image (PNG with transparency) is created on-the-fly, styled with a random color and appropriate text wrapping.
4.  **Video Overlay:** FFmpeg is used again to overlay these generated bubble images onto the original video at their designated timestamps and fixed positions, creating the final output video.

## Local Installation & Usage

To run this application locally, you'll need `ffmpeg` installed on your system.

1.  **Clone the repository:**
        ```
2.  **Make the installation script executable and run it:**
    This script will guide you through setting up a Conda or Python virtual environment and installing all necessary Python dependencies.
    ```bash
    chmod +x install.sh
    ./install.sh
    ```
    Follow the prompts, especially for PyTorch CUDA support if you have an NVIDIA GPU.
3.  **Run the application:**
    After installation completes, the script will provide instructions to activate your environment and run `app.py`. It will look something like this:
    ```bash
    # If you chose Conda
    conda activate video-bubbles
    python app.py

    # If you chose venv
    source .venv/bin/activate
    python app.py
    ```
    The application will then open in your web browser, typically at `http://127.0.0.1:7860`.

## Deployment to Hugging Face Spaces

This application is designed to be easily deployable on Hugging Face Spaces using the provided `Dockerfile`.

1.  **Create a new Space:** Go to [Hugging Face Spaces](https://huggingface.co/spaces), click "Create new Space", choose "Gradio" as SDK, and select "Docker" as the build system (this is implied when you provide a Dockerfile).
2.  **Upload Files:** Upload `app.py`, `requirements.txt`, and `Dockerfile` to your Space's repository.
3.  **Select Hardware:** Choose suitable hardware. For faster transcription with Whisper, select a GPU runtime (e.g., A10G Small). For CPU-only, choose "CPU Basic".
4.  **Build and Run:** Hugging Face will automatically build your Docker image and launch the application. You can monitor the build process in the "Logs" tab.

## Why this is valuable

While many services offer automated captioning, custom word-by-word "bubble" subtitles with dynamic styling at a fixed, clean position are less common, especially in a free, open-source, and easily deployable format. This tool offers a unique visual alternative to traditional captions, enhancing engagement and storytelling in videos.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
