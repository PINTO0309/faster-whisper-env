#! /usr/bin/env python

import io
import os
import sys
import argparse
import pyaudio
import numpy as np
import soundfile as sf
import speech_recognition as sr
from typing import BinaryIO, Iterable, List, NamedTuple, Optional, Tuple, Union, TextIO
from faster_whisper import WhisperModel

class Color:
    BLACK          = '\033[30m'
    RED            = '\033[31m'
    GREEN          = '\033[32m'
    YELLOW         = '\033[33m'
    BLUE           = '\033[34m'
    MAGENTA        = '\033[35m'
    CYAN           = '\033[36m'
    WHITE          = '\033[37m'
    COLOR_DEFAULT  = '\033[39m'
    BOLD           = '\033[1m'
    UNDERLINE      = '\033[4m'
    INVISIBLE      = '\033[08m'
    REVERCE        = '\033[07m'
    BG_BLACK       = '\033[40m'
    BG_RED         = '\033[41m'
    BG_GREEN       = '\033[42m'
    BG_YELLOW      = '\033[43m'
    BG_BLUE        = '\033[44m'
    BG_MAGENTA     = '\033[45m'
    BG_CYAN        = '\033[46m'
    BG_WHITE       = '\033[47m'
    BG_DEFAULT     = '\033[49m'
    RESET          = '\033[0m'

class Word(NamedTuple):
    start: float
    end: float
    word: str
    probability: float

class Segment(NamedTuple):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int]
    temperature: float
    avg_logprob: float
    compression_ratio: float
    no_speech_prob: float
    words: Optional[List[Word]]

def write_txt(segments: Iterable[Segment], file: TextIO):
    for segment in segments:
        print(
            f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}",
            file=file,
            flush=True,
        )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        default="audio",
        choices=["audio", "mic"],
        help="Audio file(audio) or Microphone(mic)",
    )
    parser.add_argument(
        "-a",
        "--audios",
        nargs="*",
        type=str,
        help="Specify the path to at least one or more audio files (mp4, mp3, etc.). e.g. --audio aaa.mp4 bbb.mp3 ccc.mp4",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default=".",
        help="Destination directory for transcribed text.",
    )
    parser.add_argument(
        "-s",
        "--model_size",
        type=str,
        default="large-v2",
        choices=[
            "tiny.en",
            "tiny",
            "base.en",
            "base",
            "small.en",
            "small",
            "medium.en",
            "medium",
            "large-v1",
            "large-v2",
        ],
        help="Model size.",
    )
    parser.add_argument(
        "-p",
        "--precision",
        type=str,
        default="int8_float16",
        choices=[
            "float16",
            "int8_float16",
            "int8",
        ],
        help="Model size.",
    )
    args = parser.parse_args()

    mode: str = args.mode
    output_dir: str = args.output_dir
    model_size: str = args.model_size
    precision: str = args.precision

    # Model Load
    model = None
    if precision == "float16":
        # Run on GPU with FP16
        model = WhisperModel(model_size, device="cuda", compute_type="float16")
    elif precision == "int8_float16":
        # Run on GPU with INT8
        model = WhisperModel(model_size, device="cuda", compute_type="int8_float16")
    elif precision == "int8":
        # Run on CPU with INT8
        model = WhisperModel(model_size, device="cpu", compute_type="int8")

    if mode == 'audio':
        audios: List[str] = args.audios
        if not audios:
            print(f"{Color.RED}ERROR:{Color.RESET} Specify the path to at least one or more audio files (mp4, mp3, etc.). e.g. --audio aaa.mp4 bbb.mp3 ccc.mp4")
            sys.exit(0)
        for audio_path in audios:
            segments, info = model.transcribe(audio_path, beam_size=5)
            print(f"Detected language {info.language} with probability {info.language_probability:.2f}")
            audio_basename = os.path.basename(audio_path)
            # save TXT
            with open(os.path.join(output_dir, audio_basename + ".txt"), "w", encoding="utf-8") as txt:
                write_txt(segments, file=txt)

    elif mode == 'mic':
        p = pyaudio.PyAudio()
        info = p.get_host_api_info_by_index(0)
        num_devices = info.get("deviceCount", 0)
        for i in range(num_devices):
            device_info = p.get_device_info_by_host_api_device_index(0, i)
            if device_info.get("maxInputChannels") > 0:
                print(f"Input Device ID {i}, - {device_info.get('name')}")
        device_index: int = int(input("Please input your microphone Device ID: "))
        recognizer = sr.Recognizer()
        mic = sr.Microphone(sample_rate=16_000, device_index=device_index)
        try:
            print("Speak now! (CTRL + C to exit the application)")
            while True:
                with mic as audio_source:
                    recognizer.adjust_for_ambient_noise(audio_source)
                    audio = recognizer.listen(audio_source)
                try:
                    wav_data = audio.get_wav_data()
                    wav_stream = io.BytesIO(wav_data)
                    audio_array, _ = sf.read(wav_stream)
                    audio_array = audio_array.astype(np.float32)
                    segments, info = model.transcribe(audio_array, beam_size=5)
                    for segment in segments:
                        if segment.no_speech_prob <= 0.55:
                            print(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}")
                except sr.UnknownValueError:
                    pass
                except sr.RequestError as e:
                    pass
        except KeyboardInterrupt:
            # allow CTRL + C to exit the application
            pass

if __name__ == '__main__':
    main()

