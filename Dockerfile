FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/video_labeler/requirements.txt
RUN python -m pip install --upgrade pip \
    && python -m pip install --no-cache-dir -r /app/video_labeler/requirements.txt

COPY . /app/video_labeler

EXPOSE 7860

CMD ["python", "-m", "video_labeler.main_gradio", "--host", "0.0.0.0", "--port", "7860", "--top-video-path", "/workspace/dataset/agi_arm_bot/camera_pen_touch_clean_del_52_376"]
