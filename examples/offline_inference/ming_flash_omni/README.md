# Ming-flash-omni 2.0

[Ming-flash-omni-2.0](https://github.com/inclusionAI/Ming) is an omni-modal model supporting text, image, video, and audio understanding, with outputs in text, image, and audio. For now, Ming-flash-omni-2.0 in vLLM-Omni is supported with thinker stage (multi-modal understanding).

## Setup

Please refer to the [stage configuration documentation](https://docs.vllm.ai/projects/vllm-omni/en/latest/configuration/stage_configs/) to configure memory allocation appropriately for your hardware setup.

## Run examples

### Text-only
```bash
python end2end.py --query-type text
```

### Image understanding
```bash
python end2end.py --query-type use_image

# With a local image
python end2end.py --query-type use_image --image-path /path/to/image.jpg
```

### Audio understanding
```bash
python end2end.py --query-type use_audio

# With a local audio file
python end2end.py --query-type use_audio --audio-path /path/to/audio.wav
```

### Video understanding
```bash
python end2end.py --query-type use_video

# With a local video and custom frame count
python end2end.py --query-type use_video --video-path /path/to/video.mp4 --num-frames 16
```

### Mixed modalities (image + audio)
```bash
python end2end.py --query-type use_mixed_modalities \
    --image-path /path/to/image.jpg \
    --audio-path /path/to/audio.wav
```

If media file paths are not provided, the script uses built-in default assets.

### Modality control
To control output modalities (e.g. text-only output):
```bash
python end2end.py --query-type use_audio --modalities text
```

### Custom stage config
```bash
python end2end.py --query-type use_image \
    --stage-configs-path /path/to/your_config.yaml
```
