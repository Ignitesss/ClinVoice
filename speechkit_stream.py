# -*- coding: utf-8 -*-
"""
Потоковое распознавание Yandex SpeechKit STT API v3 (gRPC).
Только для черновика в UI; финальный текст консультации — Whisper.
"""

from __future__ import annotations

import queue
import threading
from typing import Callable, List, Optional

import grpc
import yandex.cloud.ai.stt.v3.stt_pb2 as stt_pb2
import yandex.cloud.ai.stt.v3.stt_service_pb2_grpc as stt_service_pb2_grpc

STT_GRPC_TARGET = "stt.api.cloud.yandex.net:443"
TARGET_SAMPLE_RATE_HZ = 16000


def speechkit_stt_configured(api_key: Optional[str], iam_token: Optional[str]) -> bool:
    return bool((api_key or "").strip() or (iam_token or "").strip())


def build_grpc_metadata(
    api_key: Optional[str],
    iam_token: Optional[str],
    folder_id: Optional[str],
) -> List[tuple[str, str]]:
    api_key = (api_key or "").strip()
    iam_token = (iam_token or "").strip()
    folder_id = (folder_id or "").strip()
    meta: List[tuple[str, str]] = []
    if api_key:
        meta.append(("authorization", f"Api-Key {api_key}"))
    elif iam_token:
        meta.append(("authorization", f"Bearer {iam_token}"))
    else:
        raise ValueError("Нужен YANDEX_CLOUD_API_KEY или YANDEX_IAM_TOKEN для SpeechKit")
    if folder_id:
        meta.append(("x-folder-id", folder_id))
    return meta


class SpeechKitStreamRecognizer:
    """
    Один сеанс bidirectional streaming: первое сообщение — options, далее чанки PCM
    LINEAR16 mono, sample_rate как передано в конструкторе (ожидается 16000).
    """

    def __init__(
        self,
        *,
        sample_rate_hertz: int,
        api_key: Optional[str],
        iam_token: Optional[str],
        folder_id: Optional[str],
        language_codes: Optional[List[str]] = None,
        on_display_text: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[str], None]] = None,
    ) -> None:
        self._sample_rate = int(sample_rate_hertz)
        self._language_codes = language_codes or ["ru-RU"]
        self._on_display_text = on_display_text
        self._on_error = on_error
        self._meta = build_grpc_metadata(api_key, iam_token, folder_id)

        self._audio_q: "queue.Queue[Optional[bytes]]" = queue.Queue()
        self._stop = threading.Event()
        self._final_parts: List[str] = []
        self._partial: str = ""
        self._text_lock = threading.Lock()

        self._resp_thread: Optional[threading.Thread] = None
        self._channel: Optional[grpc.Channel] = None
        self._responses_iter = None

    def _emit_display(self) -> None:
        with self._text_lock:
            body = " ".join(p for p in self._final_parts if p.strip()).strip()
            tail = self._partial.strip()
            if tail:
                text = (body + " " + tail).strip() if body else tail
            else:
                text = body
        if self._on_display_text:
            self._on_display_text(text)

    def _handle_response(self, response: stt_pb2.StreamingResponse) -> None:
        event = response.WhichOneof("Event")
        texts: List[str] = []
        if event == "partial" and response.partial.alternatives:
            texts = [a.text for a in response.partial.alternatives if a.text]
        elif event == "final" and response.final.alternatives:
            texts = [a.text for a in response.final.alternatives if a.text]
        elif event == "final_refinement" and response.final_refinement.normalized_text.alternatives:
            texts = [
                a.text
                for a in response.final_refinement.normalized_text.alternatives
                if a.text
            ]

        if not texts:
            return
        primary = texts[0].strip()
        if not primary:
            return

        with self._text_lock:
            if event == "partial":
                self._partial = primary
            elif event in ("final", "final_refinement"):
                self._final_parts.append(primary)
                self._partial = ""
        self._emit_display()

    def _request_generator(self):
        recognize_options = stt_pb2.StreamingOptions(
            recognition_model=stt_pb2.RecognitionModelOptions(
                audio_format=stt_pb2.AudioFormatOptions(
                    raw_audio=stt_pb2.RawAudio(
                        audio_encoding=stt_pb2.RawAudio.LINEAR16_PCM,
                        sample_rate_hertz=self._sample_rate,
                        audio_channel_count=1,
                    )
                ),
                text_normalization=stt_pb2.TextNormalizationOptions(
                    text_normalization=stt_pb2.TextNormalizationOptions.TEXT_NORMALIZATION_ENABLED,
                    profanity_filter=True,
                    literature_text=False,
                ),
                language_restriction=stt_pb2.LanguageRestrictionOptions(
                    restriction_type=stt_pb2.LanguageRestrictionOptions.WHITELIST,
                    language_code=list(self._language_codes),
                ),
                audio_processing_type=stt_pb2.RecognitionModelOptions.REAL_TIME,
            ),
            eou_classifier=stt_pb2.EouClassifierOptions(
                default_classifier=stt_pb2.DefaultEouClassifier(
                    max_pause_between_words_hint_ms=1000
                )
            ),
        )
        yield stt_pb2.StreamingRequest(session_options=recognize_options)

        while not self._stop.is_set() or not self._audio_q.empty():
            try:
                chunk = self._audio_q.get(timeout=0.1)
            except queue.Empty:
                continue
            if chunk is None:
                break
            yield stt_pb2.StreamingRequest(chunk=stt_pb2.AudioChunk(data=chunk))
            self._audio_q.task_done()

    def _read_responses(self) -> None:
        try:
            assert self._responses_iter is not None
            for response in self._responses_iter:
                self._handle_response(response)
        except grpc.RpcError as e:
            msg = f"SpeechKit gRPC: {e.code()} {e.details() or ''}".strip()
            if self._on_error:
                self._on_error(msg)
        except Exception as e:
            if self._on_error:
                self._on_error(f"SpeechKit: {e}")
        finally:
            self._stop.set()

    def start(self) -> None:
        if self._resp_thread is not None and self._resp_thread.is_alive():
            return
        cred = grpc.ssl_channel_credentials()
        self._channel = grpc.secure_channel(STT_GRPC_TARGET, cred)
        stub = stt_service_pb2_grpc.RecognizerStub(self._channel)
        self._responses_iter = stub.RecognizeStreaming(
            self._request_generator(),
            metadata=self._meta,
        )
        self._resp_thread = threading.Thread(target=self._read_responses, daemon=True)
        self._resp_thread.start()

    def push_pcm(self, data: bytes) -> None:
        if self._stop.is_set() or not data:
            return
        self._audio_q.put(data)

    def stop(self) -> None:
        self._stop.set()
        try:
            self._audio_q.put_nowait(None)
        except queue.Full:
            pass
        if self._resp_thread and self._resp_thread.is_alive():
            self._resp_thread.join(timeout=5.0)
        if self._channel:
            try:
                self._channel.close()
            except Exception:
                pass
            self._channel = None

    def reset_text(self) -> None:
        with self._text_lock:
            self._final_parts.clear()
            self._partial = ""
        self._emit_display()


def audio_frames_to_pcm_mono_s16le_16k(
    frames: List[object],
    resampler: object,
) -> bytes:
    """PyAV AudioFrame → LINEAR16 mono 16 kHz; ``resampler`` — сохранённый AudioResampler между чанками."""
    out_chunks: List[bytes] = []
    for frame in frames:
        for converted in resampler.resample(frame):
            arr = converted.to_ndarray()
            out_chunks.append(arr.tobytes())
    return b"".join(out_chunks)


def make_audio_resampler_s16_mono_16k():
    """Создать PyAV AudioResampler (один экземпляр на сеанс WebRTC)."""
    import av

    return av.audio.resampler.AudioResampler(
        format="s16", layout="mono", rate=TARGET_SAMPLE_RATE_HZ
    )
