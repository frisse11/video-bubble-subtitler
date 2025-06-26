import gradio as gr
import os
import subprocess
from PIL import Image, ImageDraw, ImageFont
import shutil
import torch
from transformers import pipeline
import librosa
import random

# --- Configuratie ---
FONT_PATH = "/usr/share/fonts/truetype/luckiestguy/LuckiestGuy-Regular.ttf"  # Jouw lettertype pad
FOLDER_FOR_PROCESSED_VIDEOS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "processed_videos")
os.makedirs(FOLDER_FOR_PROCESSED_VIDEOS, exist_ok=True)

def create_colored_text_image(text, font_path, font_size=120, outline_width=3):
    try:
        font = ImageFont.truetype(font_path, font_size)
    except IOError:
        print(f"Waarschuwing: lettertype niet gevonden op {font_path}, gebruik default font.")
        font = ImageFont.load_default()

    bbox = font.getbbox(text)
    text_width = bbox[2] - bbox[0] + 2*outline_width
    text_height = bbox[3] - bbox[1] + 2*outline_width

    img = Image.new("RGBA", (text_width, text_height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    text_pos = (outline_width - bbox[0], outline_width - bbox[1])

    r = random.randint(150, 255)
    g = random.randint(150, 255)
    b = random.randint(150, 255)
    text_color = (r, g, b, 255)

    # Zwarte outline (stroke)
    for dx in range(-outline_width, outline_width + 1):
        for dy in range(-outline_width, outline_width + 1):
            dist = abs(dx) + abs(dy)
            if dist <= outline_width:
                draw.text((text_pos[0] + dx, text_pos[1] + dy), text, font=font, fill=(0, 0, 0, 255))

    draw.text(text_pos, text, font=font, fill=text_color)

    return img

def process_video_with_colored_words(input_video_path, font_size, vertical_offset_percent):
    if not input_video_path:
        return None, "Geen video geüpload."

    video_basename = os.path.splitext(os.path.basename(input_video_path))[0]
    final_video_output_path = os.path.join(FOLDER_FOR_PROCESSED_VIDEOS, f"{video_basename}_colored_words.mp4")
    temp_images_dir = os.path.join(FOLDER_FOR_PROCESSED_VIDEOS, f"temp_text_images_{video_basename}")
    os.makedirs(temp_images_dir, exist_ok=True)

    words_with_timestamps = []

    # --- Audio extractie ---
    audio_path = os.path.join(FOLDER_FOR_PROCESSED_VIDEOS, f"{video_basename}_audio.wav")
    try:
        subprocess.run([
            "ffmpeg",
            "-i", input_video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            "-y", audio_path
        ], check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        if os.path.exists(temp_images_dir):
            shutil.rmtree(temp_images_dir)
        return None, f"Fout bij audio extractie: {e.stderr}"

    # --- Whisper transcriptie ---
    try:
        device = 0 if torch.cuda.is_available() else -1
        pipe = pipeline(
            "automatic-speech-recognition",
            model="openai/whisper-small",
            return_timestamps="word",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device=device,
            chunk_length_s=30,
            stride_length_s=(5, 5)
        )
        audio_input, sr = librosa.load(audio_path, sr=16000, mono=True)
        result = pipe(audio_input)

        if 'chunks' not in result or not result['chunks']:
            return None, "Geen woorden met timestamps gevonden."

        for word_info in result['chunks']:
            word_text = word_info.get('text', '').strip()
            timestamp = word_info.get('timestamp')
            if timestamp and len(timestamp) == 2:
                start_sec = timestamp[0]
                end_sec = timestamp[1]
            else:
                start_sec = None
                end_sec = None
            if word_text and start_sec is not None and end_sec is not None and end_sec > start_sec:
                words_with_timestamps.append({
                    'word': word_text,
                    'start': float(start_sec),
                    'end': float(end_sec)
                })
    except Exception as e:
        if os.path.exists(temp_images_dir):
            shutil.rmtree(temp_images_dir)
        if os.path.exists(audio_path):
            os.remove(audio_path)
        return None, f"Fout bij transcriptie: {str(e)}"

    if not words_with_timestamps:
        return None, "Geen woorden met timestamps gevonden."

    # --- FFmpeg filter en inputs ---
    ffmpeg_filters = []
    image_inputs = []

    current_video_stream = "base_video"
    ffmpeg_filters.append(f"[0:v]format=yuv420p[base_video]")

    # Posities: horizontaal gecentreerd, verticaal via slider (% hoogte vanaf onderkant)
    x_pos = "(W - w)/2"
    # vertical_offset_percent is van 0 (helemaal onder) tot 100 (helemaal boven)
    # We plaatsen tekst op: H - h - (H * vertical_offset_percent/100)
    y_pos = f"H - h - H*{vertical_offset_percent/100:.3f}"

    for i, word_data in enumerate(words_with_timestamps):
        word_text = word_data['word']
        start_sec = word_data['start']
        end_sec = word_data['end']

        if not word_text or end_sec <= start_sec:
            continue

        img = create_colored_text_image(word_text, FONT_PATH, font_size=font_size, outline_width=3)
        img_path = os.path.join(temp_images_dir, f"text_{i:04d}.png")
        img.save(img_path)

        image_inputs.extend(["-i", img_path])

        filter_str = f"[{current_video_stream}][{i+1}:v]overlay={x_pos}:{y_pos}:enable='between(t,{start_sec},{end_sec})'[v{i+1}]"
        ffmpeg_filters.append(filter_str)
        current_video_stream = f"v{i+1}"

    ffmpeg_inputs = ["-i", input_video_path] + image_inputs
    full_filter_complex = ";".join(ffmpeg_filters)

    video_output_stream_to_map = f"[{current_video_stream}]" if ffmpeg_filters else "[base_video]"

    ffmpeg_cmd = [
        "ffmpeg",
        *ffmpeg_inputs,
        "-filter_complex", full_filter_complex,
        "-c:v", "libx264",
        "-preset", "medium",
        "-crf", "23",
        "-map", video_output_stream_to_map,
        "-map", "0:a",
        "-y",
        final_video_output_path
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True, text=True)
        return final_video_output_path, "Video succesvol verwerkt!"
    except subprocess.CalledProcessError as e:
        return None, f"Fout bij video overlay: {e.stderr}"
    finally:
        if os.path.exists(temp_images_dir):
            shutil.rmtree(temp_images_dir)
        if os.path.exists(audio_path):
            os.remove(audio_path)

if __name__ == "__main__":
    if not os.path.exists(FONT_PATH):
        print(f"Lettertype niet gevonden: {FONT_PATH}. Pas FONT_PATH aan.")
    
    with gr.Blocks() as demo:
        gr.Markdown("# Video Subtitler met Kleurrijke Woorden (zonder bubbels)")

        with gr.Row():
            with gr.Column(scale=1):
                video_input = gr.File(label="Upload video", type="filepath", file_types=["video"])
                font_size_slider = gr.Slider(label="Fontgrootte", minimum=40, maximum=200, value=120, step=1)
                vertical_offset_slider = gr.Slider(label="Verticale positie (%) boven onderkant", minimum=0, maximum=100, value=12, step=1)
                process_button = gr.Button("Verwerk video")
                status_output = gr.Textbox(label="Status", interactive=False, lines=3)
            
            with gr.Column(scale=2):
                output_video = gr.Video(label="Verwerkte video", height=480)

        process_button.click(
            fn=process_video_with_colored_words,
            inputs=[video_input, font_size_slider, vertical_offset_slider],
            outputs=[output_video, status_output]
        )

    demo.launch(share=False)

