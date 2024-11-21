import queue
import re
import sys
import time
import io
import threading

from google.cloud import speech, texttospeech
import pyaudio
from pydub import AudioSegment
from pydub.playback import play

# 오디오 녹음 설정
RATE = 16000
CHUNK = int(RATE / 10)  # 100ms

# 전역 변수로 모델의 발화 상태를 추적
is_speaking = False

class MicrophoneStream:
    def __init__(self, rate=RATE, chunk=CHUNK):
        self._rate = rate
        self._chunk = chunk
        self._buff = queue.Queue()
        self.closed = True
        self._audio_interface = None
        self._audio_stream = None

    def __enter__(self):
        self._audio_interface = pyaudio.PyAudio()
        self.closed = False
        self._audio_stream = self._audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=self._rate,
            input=True,
            frames_per_buffer=self._chunk,
            stream_callback=self._fill_buffer,
        )
        return self

    def __exit__(self, type, value, traceback):
        if self._audio_stream is not None:
            self._audio_stream.stop_stream()
            self._audio_stream.close()
        self.closed = True
        self._buff.put(None)
        if self._audio_interface is not None:
            self._audio_interface.terminate()

    def pause(self):
        if self._audio_stream is not None and self._audio_stream.is_active():
            self._audio_stream.stop_stream()
        self.closed = True

    def resume(self):
        if self._audio_stream is not None and not self._audio_stream.is_active():
            self._audio_stream.start_stream()
        self.closed = False
        # 버퍼 비우기
        while not self._buff.empty():
            self._buff.get()

    def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
        if not self.closed:
            self._buff.put(in_data)
        return None, pyaudio.paContinue

    def generator(self):
        while not self.closed:
            try:
                chunk = self._buff.get(timeout=1)
            except queue.Empty:
                continue
            if chunk is None:
                return
            data = [chunk]

            while True:
                try:
                    chunk = self._buff.get(block=False)
                    if chunk is None:
                        return
                    data.append(chunk)
                except queue.Empty:
                    break

            yield b''.join(data)

def generate_speech(text):
    """Google Text-to-Speech API를 사용하여 음성 생성."""
    global is_speaking
    is_speaking = True  # 발화 시작
    client = texttospeech.TextToSpeechClient()

    input_text = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code="ko-KR",
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        input=input_text,
        voice=voice,
        audio_config=audio_config
    )

    # 메모리 내 버퍼를 사용하여 오디오 재생
    audio_data = io.BytesIO(response.audio_content)
    audio_segment = AudioSegment.from_file(audio_data, format="mp3")

    # 오디오 재생
    play(audio_segment)
    is_speaking = False  # 발화 종료

def delayed_response(delay, message, stream):
    """지정된 시간 후에 음성 메시지를 생성하고 마이크 입력을 제어."""
    time.sleep(delay)
    stream.pause()
    generate_speech(message)
    stream.resume()

def listen_print_loop(responses, stream):
    global is_speaking
    num_chars_printed = 0
    for response in responses:
        if not response.results:
            continue

        # 첫 번째 결과 가져오기
        result = response.results[0]
        if not result.alternatives:
            continue

        # 인식된 텍스트 추출
        transcript = result.alternatives[0].transcript.strip()
        overwrite_chars = ' ' * (num_chars_printed - len(transcript))

        if not result.is_final:
            if not is_speaking:
                # 실시간 텍스트 출력
                sys.stdout.write(transcript + overwrite_chars + '\r')
                sys.stdout.flush()
                num_chars_printed = len(transcript)
        else:
            if not is_speaking:
                # 최종 인식 결과 출력
                print(transcript + overwrite_chars)

                if re.search(r'\b(종료|스탑)\b', transcript, re.I):
                    print('프로그램을 종료합니다.')
                    sys.exit(0)  # 프로그램 완전 종료

                # 특정 단어에 따른 처리
                if '갱킹' in transcript:
                    message = f"{transcript} 접수!"
                    threading.Thread(target=delayed_response, args=(1, message, stream)).start()
                    message = f"{transcript} 30초가 지났습니다."
                    threading.Thread(target=delayed_response, args=(3, message, stream)).start()
                elif '플' in transcript:
                    message = f"{transcript} 접수!"
                    threading.Thread(target=delayed_response, args=(1, message, stream)).start()
                    message = f"{transcript} 30초가 지났습니다."
                    threading.Thread(target=delayed_response, args=(3, message, stream)).start()
                else:
                    # 다른 경우 필요에 따라 처리
                    pass

            num_chars_printed = 0

def main():
    language_code = "ko-KR"  # 한국어

    client = speech.SpeechClient()
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=RATE,
        language_code=language_code,
    )
    streaming_config = speech.StreamingRecognitionConfig(
        config=config,
        interim_results=True,
        single_utterance=False  # 연속적인 인식을 위해 False로 설정
    )

    with MicrophoneStream(RATE, CHUNK) as stream:
        while True:
            try:
                audio_generator = stream.generator()
                requests = (
                    speech.StreamingRecognizeRequest(audio_content=content)
                    for content in audio_generator
                )

                responses = client.streaming_recognize(streaming_config, requests)

                # listen_print_loop 함수에 stream 전달
                listen_print_loop(responses, stream)
            except Exception as e:
                print(f"오류 발생: {e}")
                # 잠시 대기 후 스트림 재시작
                time.sleep(1)

if __name__ == '__main__':
    main()




# import queue
# import re
# import sys
# import time
# import io

# from google.cloud import speech, texttospeech
# import pyaudio
# from pydub import AudioSegment
# from pydub.playback import play

# # Audio recording parameters
# RATE = 16000
# CHUNK = int(RATE / 10)  # 100ms

# class MicrophoneStream:
#     def __init__(self, rate=RATE, chunk=CHUNK):
#         self._rate = rate
#         self._chunk = chunk
#         self._buff = queue.Queue()
#         self.closed = True

#     def __enter__(self):
#         self._audio_interface = pyaudio.PyAudio()
#         self._audio_stream = self._audio_interface.open(
#             format=pyaudio.paInt16,
#             channels=1,
#             rate=self._rate,
#             input=True,
#             frames_per_buffer=self._chunk,
#             stream_callback=self._fill_buffer,
#         )
#         self.closed = False
#         return self

#     def __exit__(self, type, value, traceback):
#         self._audio_stream.stop_stream()
#         self._audio_stream.close()
#         self.closed = True
#         self._buff.put(None)
#         self._audio_interface.terminate()

#     def _fill_buffer(self, in_data, frame_count, time_info, status_flags):
#         self._buff.put(in_data)
#         return None, pyaudio.paContinue

#     def generator(self):
#         while not self.closed:
#             chunk = self._buff.get()
#             if chunk is None:
#                 return
#             data = [chunk]
#             while True:
#                 try:
#                     chunk = self._buff.get(block=False)
#                     if chunk is None:
#                         return
#                     data.append(chunk)
#                 except queue.Empty:
#                     break
#             yield b''.join(data)

# def listen_print_loop(responses):
#     num_chars_printed = 0
#     for response in responses:
#         if not response.results:
#             continue

#         result = response.results[0]
#         if not result.alternatives:
#             continue

#         transcript = result.alternatives[0].transcript
#         overwrite_chars = ' ' * (num_chars_printed - len(transcript))

#         if not result.is_final:
#             sys.stdout.write(transcript + overwrite_chars + '\r')
#             sys.stdout.flush()
#             num_chars_printed = len(transcript)
#         else:
#             print(transcript + overwrite_chars)

#             if re.search(r'\b(exit|quit)\b', transcript, re.I):
#                 print('Exiting..')
#                 break

#             # Step 1: Wait for 30 seconds after the initial speech
#             time.sleep(5)
#             # Step 2: Generate speech after the delay
#             generate_speech(f"{transcript} 후 30초가 지났습니다.")
#             num_chars_printed = 0

#     return transcript

# def generate_speech(text):
#     """Generate speech from text using Google Text-to-Speech API."""
#     client = texttospeech.TextToSpeechClient()

#     input_text = texttospeech.SynthesisInput(text=text)
#     voice = texttospeech.VoiceSelectionParams(
#         language_code="ko-KR", ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
#     )
#     audio_config = texttospeech.AudioConfig(
#         audio_encoding=texttospeech.AudioEncoding.MP3
#     )

#     response = client.synthesize_speech(input=input_text, voice=voice, audio_config=audio_config)

#     # Use in-memory buffer instead of saving to file
#     audio_data = io.BytesIO(response.audio_content)
#     audio_segment = AudioSegment.from_file(audio_data, format="mp3")
    
#     # Play the audio
#     play(audio_segment)

# def main():
#     language_code = "ko-KR"  # Korean

#     client = speech.SpeechClient()
#     config = speech.RecognitionConfig(
#         encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
#         sample_rate_hertz=RATE,
#         language_code=language_code,
#     )
#     streaming_config = speech.StreamingRecognitionConfig(
#         config=config, interim_results=True
#     )

#     with MicrophoneStream(RATE, CHUNK) as stream:
#         audio_generator = stream.generator()
#         requests = (
#             speech.StreamingRecognizeRequest(audio_content=content)
#             for content in audio_generator
#         )

#         responses = client.streaming_recognize(streaming_config, requests)

#         listen_print_loop(responses)

# if __name__ == '__main__':
#     main()



