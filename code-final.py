# This python script was created by Marc Fabry in the year 2025.Tested on linux mint, but probably works on multiple systems. The author cannot be held responsible for any damage and or errors that occur when using this script.
# Please use it responsible ,share and improve.

import gradio as gr
import os
import subprocess
from PIL import Image, ImageDraw, ImageFont
import shutil
import textwrap
import torch
from transformers import pipeline
import librosa
import random 

# --- Configuration (adjust if necessary) ---
# Important: Ensure this path points to a valid TrueType font file on your system.
# For Linux (Debian/Ubuntu-based), these are common default paths.
# If you are using a different distribution or OS, adjust this path. E.g.:
# FONT_PATH = "/System/Library/Fonts/SFCompactText-Bold.otf" # macOS example
# FONT_PATH = "C:/Windows/Fonts/arialbd.ttf" # Windows example
FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

FOLDER_FOR_PROCESSED_VIDEOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_videos")

# Ensure the output directory exists
os.makedirs(FOLDER_FOR_PROCESSED_VIDEOS, exist_ok=True)

# --- Function to create bubble image ---
def create_bubble_image(text, font_path, font_size=40, text_color=(0, 0, 0), bubble_color=None, padding=20, max_width_pixels=600):
    """
    Generates a semi-transparent bubble image with the given text.
    The bubble has rounded corners and a small "tail" at the bottom.
    The bubble and text colors are chosen randomly if not specified,
    and the text color is adjusted for readability.
    """
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Warning: Font not found at {font_path}. Using default font.")
        font = ImageFont.load_default() # Fallback to a default font

    # Generate a random bright color for the bubble if none is specified
    if bubble_color is None:
        r = random.randint(150, 255) 
        g = random.randint(150, 255)
        b = random.randint(150, 255)
        a = 200 # Semi-transparent
        bubble_color = (r, g, b, a)

        # Adjust text color based on bubble color brightness
        # Dark text on light bubble, light text on dark bubble
        avg_brightness = (r + g + b) / 3
        if avg_brightness < 128: 
            text_color = (255, 255, 255) # White text
        else:
            text_color = (0, 0, 0) # Black text

    # Determine the maximum number of characters per line to wrap text
    # Use an average width of 'm' to estimate
    test_string = "m" * 10
    # getbbox() returns (left, top, right, bottom) of the text
    test_width = font.getbbox(test_string)[2] - font.getbbox(test_string)[0]
    avg_char_width = test_width / 10 if test_width > 0 else font_size * 0.6
    
    max_chars_per_line = int((max_width_pixels - 2 * padding) / avg_char_width)
    if max_chars_per_line < 1: # Ensure at least 1 character fits per line
        max_chars_per_line = 1 

    # Wrap the text to fit within the maximum width
    wrapped_text_lines = textwrap.wrap(text, width=max_chars_per_line)
    
    total_text_width = 0
    # Calculate line height for accurate bubble height
    line_height = font.getbbox("Ay")[3] - font.getbbox("Ay")[1] 
    total_text_height = len(wrapped_text_lines) * line_height

    # Determine the widest line
    for line in wrapped_text_lines:
        line_bbox = font.getbbox(line)
        total_text_width = max(total_text_width, line_bbox[2] - line_bbox[0])

    # Calculate the final bubble dimensions
    bubble_width = total_text_width + 2 * padding
    bubble_height = total_text_height + 2 * padding + 10 # +10 for the tail

    min_bubble_width = 100 # Minimum width for small words
    bubble_width = max(bubble_width, min_bubble_width)

    # Create a transparent image
    img = Image.new('RGBA', (bubble_width, bubble_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # Draw the rounded rectangle for the bubble body
    radius = 15
    draw.rounded_rectangle([(0, 0), (bubble_width, bubble_height - 10)], radius=radius, fill=bubble_color)

    # Draw the bubble tail (triangle)
    tail_base_x = bubble_width // 2 - 10 # Base of the tail
    tail_base_y = bubble_height - 10
    draw.polygon([(tail_base_x, tail_base_y),
                  (tail_base_x + 20, tail_base_y),
                  (bubble_width // 2, bubble_height)], # Tip of the tail
                 fill=bubble_color)

    # Place the wrapped text inside the bubble
    y_text = padding
    for line in wrapped_text_lines:
        line_bbox = font.getbbox(line)
        # Center the text per line
        line_x_offset = (bubble_width - (line_bbox[2] - line_bbox[0])) / 2 
        draw.text((line_x_offset, y_text), line, font=font, fill=text_color)
        y_text += line_height

    return img

# --- Main function for video processing ---
def process_video_with_bubbles(input_video_path):
    """
    Processes an input video: extracts audio, transcribes it with Whisper,
    generates bubble images for each word, and overlays them onto the video
    using FFmpeg at a fixed position.
    """
    if not input_video_path:
        return None, "No video file uploaded."

    video_basename = os.path.splitext(os.path.basename(input_video_path))[0]
    
    final_video_output_path = os.path.join(FOLDER_FOR_PROCESSED_VIDEOS, f"{video_basename}_bubbled.mp4")
    temp_images_dir = os.path.join(FOLDER_FOR_PROCESSED_VIDEOS, f"temp_bubble_images_{video_basename}")
    
    os.makedirs(temp_images_dir, exist_ok=True)

    words_with_timestamps = []

    # --- Step 1: Extract audio from the video ---
    print(f"Extracting audio from: {input_video_path}")
    audio_path = os.path.join(FOLDER_FOR_PROCESSED_VIDEOS, f"{video_basename}_audio.wav")
    try:
        subprocess.run([
            "ffmpeg",
            "-i", input_video_path,
            "-vn", # No video stream
            "-acodec", "pcm_s16le", # Audio codec (uncompressed)
            "-ar", "16000", # Sample rate
            "-ac", "1", # Mono audio
            "-y", audio_path # Overwrite if exists
        ], check=True, capture_output=True, text=True)
        print(f"Audio successfully extracted to: {audio_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error during audio extraction: {e.stderr}")
        if os.path.exists(temp_images_dir):
            shutil.rmtree(temp_images_dir)
        return None, f"Error during audio extraction: {e.stderr}"

    # --- Step 2: Transcribe audio with Hugging Face Transformers (Whisper) ---
    print(f"Starting Whisper transcription for: {audio_path}")
    try:
        # Detect if a GPU (CUDA) is available
        device = 0 if torch.cuda.is_available() else -1
        print(f"Using device: {'GPU (cuda)' if device == 0 else 'CPU'}")

        # Load the Whisper model and configure for word-level timestamps
        pipe = pipeline(
            "automatic-speech-recognition", 
            model="openai/whisper-small", # Choose a model (small, base, medium, large)
            return_timestamps="word", # Request word-level timestamps
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32, 
            device=device,
            chunk_length_s=30, # Process in chunks for memory management
            stride_length_s=(5, 5) # Overlap chunks to maintain context
        )
        
        # Load audio with librosa to the required sample rate
        print(f"Loading audio with librosa: {audio_path}")
        audio_input, sr = librosa.load(audio_path, sr=16000, mono=True)
        print(f"Audio loaded: sample rate={sr}, length={len(audio_input)} samples, duration={len(audio_input)/sr:.2f} seconds")

        # Perform the transcription
        result = pipe(audio_input) 

        if 'chunks' not in result or not result['chunks']:
            print("Whisper output contains no usable 'chunks' to process.")
            return None, "No words with timestamps found in Whisper output. Check audio or model."

        # Parse the results and store words with timestamps
        for word_info in result['chunks']:
            word_text = word_info.get('text', '').strip()
            timestamp = word_info.get('timestamp')

            if timestamp and len(timestamp) == 2:
                start_sec = timestamp[0]
                end_sec = timestamp[1]
            else:
                start_sec = None
                end_sec = None

            # Check for valid words and timestamps
            if word_text and start_sec is not None and end_sec is not None and end_sec > start_sec:
                words_with_timestamps.append({
                    'word': word_text,
                    'start': float(start_sec),
                    'end': float(end_sec)
                })
            else:
                print(f"Warning: Invalid word/timestamp found: {word_info} - skipping.")
        
        if not words_with_timestamps:
            return None, "No words with timestamps found in Whisper output. Check audio or model."
        
        print(f"Whisper transcription completed: {len(words_with_timestamps)} words found.")

    except Exception as e:
        print(f"Error during Whisper transcription: {e}")
        import traceback
        traceback.print_exc() # Print full traceback for debugging
        if os.path.exists(temp_images_dir):
            shutil.rmtree(temp_images_dir)
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return None, f"Error during transcription: {e}"

    # --- Step 3: Generate bubble images and build FFmpeg filter string (FIXED POSITION) ---
    ffmpeg_filters = []
    image_inputs = [] 
    
    current_video_stream = "base_video" 
    # Format the video to yuv420p for broad compatibility
    ffmpeg_filters.append(f"[0:v]format=yuv420p[base_video]") 

    # --- DEFINE THE FIXED POSITION HERE ---
    # X-position: Center of the screen
    # (W - w)/2 means: (Video_Width - Bubble_Width) / 2
    x_pos = "(W - w)/2" 

    # Y-position: 12% of total height from the bottom edge
    # H is total video height
    # h is bubble height
    # H - h places the bubble exactly on the bottom edge (top of bubble = bottom of video - bubble height)
    # H*0.12 is 12% of total height
    # H - h - H*0.12 moves the bubble 12% of the total video height ABOVE the bottom edge.
    y_pos = "H - h - H*0.12" 
    # --- END FIXED POSITION DEFINITION ---


    for i, word_data in enumerate(words_with_timestamps):
        word_text = word_data['word']
        start_sec = word_data['start']
        end_sec = word_data['end']

        if not word_text or end_sec <= start_sec:
            print(f"Warning: Word '{word_text}' with duration ({start_sec}-{end_sec}) is invalid, skipping.")
            continue

        bubble_img = create_bubble_image(word_text, font_path=FONT_PATH)
        img_path = os.path.join(temp_images_dir, f"bubble_{i:04d}.png")
        bubble_img.save(img_path)

        # Add the generated image as input for FFmpeg
        image_inputs.extend(["-i", img_path])

        # Use the FIXED x_pos and y_pos defined above
        # [previous_stream][bubble_input]overlay=x:y:enable='time_condition'[new_stream]
        filter_str = f"[{current_video_stream}][{i+1}:v]overlay={x_pos}:{y_pos}:enable='between(t,{start_sec},{end_sec})'[v{i+1}]"
        ffmpeg_filters.append(filter_str)
        current_video_stream = f"v{i+1}" # The output of this filter becomes the input for the next
    
    # Combine all inputs and filters for the FFmpeg command
    ffmpeg_inputs = ["-i", input_video_path] + image_inputs
    full_filter_complex = ";".join(ffmpeg_filters)

    # Determine which video stream should be the final output
    if not ffmpeg_filters: # If there are no bubbles, use the base_video stream
        video_output_stream_to_map = "[base_video]" 
    else: # Otherwise, use the last created stream (the last vX)
        video_output_stream_to_map = f"[{current_video_stream}]" 

    # Build the FFmpeg command
    ffmpeg_cmd = [
        "ffmpeg",
        *ffmpeg_inputs,
    ]

    if ffmpeg_filters: # Only add if filters actually exist
        ffmpeg_cmd.append("-filter_complex")
        ffmpeg_cmd.append(full_filter_complex)

    ffmpeg_cmd.extend([
        "-c:v", "libx264", # Video codec
        "-preset", "medium", # Encoding speed/quality tradeoff
        "-crf", "23", # Constant Rate Factor (quality: 0-51, lower is better)
        "-map", video_output_stream_to_map, # Map the final video stream
        "-map", "0:a", # Map the original audio stream from the input video
        "-y", # Overwrite output file without prompting
        final_video_output_path
    ])

    print("Executing FFmpeg command:")
    print("FFmpeg Command:", " ".join(ffmpeg_cmd))
    print("Filter Complex:", full_filter_complex)
    print("ffmpeg_filters list:", ffmpeg_filters)
    print("video_output_stream_to_map:", video_output_stream_to_map)
    
    # --- Step 4: Execute FFmpeg and clean up ---
    try:
        # Execute FFmpeg and capture output for debugging
        process = subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
        print(f"Video with bubbles successfully saved as: {final_video_output_path}")
        return final_video_output_path, "Video successfully processed!"
    except subprocess.CalledProcessError as e:
        print(f"Error during FFmpeg: {e.stderr}")
        return None, f"Error during video overlay: {e.stderr}"
    finally:
        # Clean up temporary files
        if os.path.exists(temp_images_dir):
            shutil.rmtree(temp_images_dir)
            print(f"Temporary directory '{temp_images_dir}' cleaned up.")
        if os.path.exists(audio_path):
            os.remove(audio_path) 
            print(f"Temporary audio '{audio_path}' cleaned up.")

# --- Gradio Interface Definition ---
if __name__ == "__main__":
    # Check if the font path exists on startup
    if not os.path.exists(FONT_PATH):
        print(f"Error: The specified font path does not exist: {FONT_PATH}")
        print("Ensure that FONT_PATH in app.py points to a valid TrueType font file on your system.")
        # Attempt to find a more general DejaVu Sans if the Bold version is missing
        fallback_font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        if os.path.exists(fallback_font_path):
            FONT_PATH = fallback_font_path
            print(f"Using fallback font instead: {FONT_PATH}")
        else:
            print("No known fallback font found. Text rendering may cause issues.")

    with gr.Blocks() as demo:
        gr.Markdown(
            """
            # Video Bubble Subtitler
            Upload a video (MP4, MOV, etc.) and get back a version with automatic 
            word-by-word bubble subtitles at a fixed position.
            """
        )
        
        # Use gr.Row to structure the layout
        with gr.Row():
            # Column for input, button, and status message
            with gr.Column(scale=1): # This column takes 1 portion of the available width
                video_input = gr.File(label="Upload your video", type="filepath", file_types=["video"])
                process_button = gr.Button("Generate Bubbles")
                status_output = gr.Textbox(label="Status", interactive=False, lines=3)
            
            # Column for the processed video preview
            with gr.Column(scale=2): # This column takes 2 portions of the available width (i.e., 2x wider than the input column)
                # IMPORTANT CHANGE HERE: 'height' parameter added to gr.Video
                output_video = gr.Video(label="Processed Video", show_label=True, height=480) 
                # You can adjust 'height' to another value (e.g., 360, 640) 
                # depending on how large you want the video window to be.
                # If you want it to adapt to width, you can try 'width="100%"' and 'height="auto"' 
                # but the 'height' parameter works more directly for a fixed max-height.

        # Link the button to the processing function
        process_button.click(
            fn=process_video_with_bubbles,
            inputs=[video_input],
            outputs=[output_video, status_output]
        )

    # Start the Gradio interface
    demo.launch(share=False) # share=True to generate a public link (useful for testing)
