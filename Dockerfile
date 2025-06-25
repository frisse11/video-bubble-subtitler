# Use a Python 3.10 slim image based on Debian Bullseye (common for HF Spaces)
FROM python:3.10-slim-bullseye

# Set the working directory inside the container
WORKDIR /app

# Install system dependencies:
# - ffmpeg: Essential for video/audio processing.
# - fonts-dejavu-core: Provides the DejaVu fonts, including DejaVuSans-Bold.ttf, at /usr/share/fonts/truetype/dejavu/.
# - git: Useful if you need to clone other repos inside the Space (e.g., for custom models).
# --no-install-recommends: To keep the image size smaller.
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    ffmpeg \
    fonts-dejavu-core \
    git \
    && \
    rm -rf /var/lib/apt/lists/*

# Copy the requirements.txt file into the container
COPY requirements.txt .

# Install Python dependencies from requirements.txt
# --no-cache-dir: To prevent pip from storing cache, reducing image size.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of your application code into the container
COPY . .

# Expose the port on which Gradio applications typically run
EXPOSE 7860

# Define the command to run your Gradio application when the container starts
CMD ["python", "app.py"]