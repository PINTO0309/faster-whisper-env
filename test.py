#! /usr/bin/env python

from faster_whisper import WhisperModel
from performance_monitor import *

# model_size = "large-v2"
model_size = "large-v3"

# Run on GPU with FP16
model = WhisperModel(model_size, device="cuda", compute_type="float16")

# or run on GPU with INT8
# model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
# or run on CPU with INT8
# model = WhisperModel(model_size, device="cpu", compute_type="int8")

segments, info = model.transcribe("testGGG_mic.wav", beam_size=5)
print(f"Detected language {info.language} with probability {info.language_probability:.2f}")

@performance_sec
def main():
    for segment in segments:
        print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")

if __name__ == '__main__':
    main()