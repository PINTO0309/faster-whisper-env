#! /usr/bin/env python

import asyncio
import argparse
import pyaudio
import queue
import numpy as np
import webrtcvad
from concurrent.futures import ThreadPoolExecutor
from typing import List, NamedTuple, Optional
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


### Cited: https://github.com/reriiasu/speech-to-text
def create_audio_stream(selected_device_index, callback):
    RATE = 16000
    CHUNK = 480
    FORMAT = pyaudio.paInt16
    CHANNELS = 1

    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=FORMAT,
        channels=CHANNELS,
        rate=RATE,
        input=True,
        input_device_index=selected_device_index,
        frames_per_buffer=CHUNK,
        stream_callback=callback,
    )

    return stream

### Cited: https://github.com/reriiasu/speech-to-text
class VadWrapper:
    def __init__(self):
        self.vad = webrtcvad.Vad(3)
        self.RATE = 16000
        self.SILENT_CHUNKS_THRESHOLD = 20

    def is_speech(self, in_data):
        return self.vad.is_speech(in_data, self.RATE)

### Cited: https://github.com/reriiasu/speech-to-text
class WhisperModelWrapper:
    def __init__(self, model_size="large-v2", precision="int8_float16", language="ja"):
        self.model_size_or_path = model_size
        self.language = language
        if precision == "float16":
            # Run on GPU with FP16
            self.model = WhisperModel(
                self.model_size_or_path, device="cuda", compute_type="float16"
            )
        elif precision == "int8_float16":
            # Run on GPU with INT8
            self.model = WhisperModel(
                self.model_size_or_path, device="cuda", compute_type="int8_float16"
            )
        elif precision == "int8":
            # Run on CPU with INT8
            self.model = WhisperModel(
                self.model_size_or_path, device="cpu", compute_type="int8"
            )

    def transcribe(self, audio):
        segments, _ = self.model.transcribe(
            audio=audio, beam_size=5, language=self.language, without_timestamps=True
        )
        return segments

### Cited: https://github.com/reriiasu/speech-to-text
class AudioTranscriber:
    def __init__(self, model_size="large-v2", precision="int8_float16", language="ja"):
        self.model_wrapper = WhisperModelWrapper(model_size=model_size, precision=precision, language=language)
        self.vad_wrapper = VadWrapper()
        self.silent_chunks = 0
        self.speech_buffer = []
        self.audio_queue = queue.Queue()

    async def transcribe_audio(self):
        with ThreadPoolExecutor() as executor:
            while True:
                audio_data_np = await asyncio.get_event_loop().run_in_executor(
                    executor, self.audio_queue.get
                )
                segments = await asyncio.get_event_loop().run_in_executor(
                    executor, self.model_wrapper.transcribe, audio_data_np
                )

                for segment in segments:
                    print(segment.text)

    def process_audio(self, in_data, frame_count, time_info, status):
        is_speech = self.vad_wrapper.is_speech(in_data)

        if is_speech:
            self.silent_chunks = 0
            audio_data = np.frombuffer(in_data, dtype=np.int16)
            self.speech_buffer.append(audio_data)
        else:
            self.silent_chunks += 1

        if (
            not is_speech
            and self.silent_chunks > self.vad_wrapper.SILENT_CHUNKS_THRESHOLD
        ):
            if len(self.speech_buffer) > 20:
                audio_data_np = np.concatenate(self.speech_buffer)
                self.speech_buffer.clear()
                self.audio_queue.put(audio_data_np)
            else:
                # noise clear
                self.speech_buffer.clear()

        return (in_data, pyaudio.paContinue)

    def start_transcription(self, selected_device_index):
        stream = create_audio_stream(selected_device_index, self.process_audio)
        print("Speak now!")
        asyncio.run(self.transcribe_audio())
        stream.start_stream()
        try:
            while True:
                key = input("Press Enter to exit.\n")
                if not key:
                    break
        except KeyboardInterrupt:
            print("Interrupted.")
        finally:
            stream.stop_stream()
            stream.close()


def main():
    parser = argparse.ArgumentParser()
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
        help="Precision.",
    )
    parser.add_argument(
        "-l",
        "--language",
        type=str,
        default="ja",
        choices=[
            "en",
            "zh",
            "de",
            "es",
            "ru",
            "ko",
            "fr",
            "ja",
            "pt",
            "tr",
            "pl",
            "ca",
            "nl",
            "ar",
            "sv",
            "it",
            "id",
            "hi",
            "fi",
            "vi",
            "iw",
            "uk",
            "el",
            "ms",
            "cs",
            "ro",
            "da",
            "hu",
            "ta",
            "no",
            "th",
            "ur",
            "hr",
            "bg",
            "lt",
            "la",
            "mi",
            "ml",
            "cy",
            "sk",
            "te",
            "fa",
            "lv",
            "bn",
            "sr",
            "az",
            "sl",
            "kn",
            "et",
            "mk",
            "br",
            "eu",
            "is",
            "hy",
            "ne",
            "mn",
            "bs",
            "kk",
            "sq",
            "sw",
            "gl",
            "mr",
            "pa",
            "si",
            "km",
            "sn",
            "yo",
            "so",
            "af",
            "oc",
            "ka",
            "be",
            "tg",
            "sd",
            "gu",
            "am",
            "yi",
            "lo",
            "uz",
            "fo",
            "ht",
            "ps",
            "tk",
            "nn",
            "mt",
            "sa",
            "lb",
            "my",
            "bo",
            "tl",
            "mg",
            "as",
            "tt",
            "haw",
            "ln",
            "ha",
            "ba",
            "jw",
            "su",
        ],
        help="Precision.",
    )
    args = parser.parse_args()

    model_size: str = args.model_size
    precision: str = args.precision
    language: str = args.language

    p = pyaudio.PyAudio()
    info = p.get_host_api_info_by_index(0)
    num_devices = info.get("deviceCount", 0)
    for i in range(num_devices):
        device_info = p.get_device_info_by_host_api_device_index(0, i)
        if device_info.get("maxInputChannels") > 0:
            print(f"Input Device ID {i}, - {device_info.get('name')}")
    device_index: int = int(input("Please input your microphone Device ID: "))
    transcriber = AudioTranscriber(model_size=model_size, precision=precision, language=language)
    transcriber.start_transcription(device_index)


if __name__ == '__main__':
    main()

