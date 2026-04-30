# -*- coding: utf-8 -*-
"""Whisper / faster-whisper / transformers ASR без Streamlit."""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import wave
from typing import TYPE_CHECKING, List, Optional

import torch
import whisper

from clinvoice_audio_utils import pcm_mono_s16le_to_wav_bytes
from clinvoice_cache import resolve_app_cache_root

if TYPE_CHECKING:
    pass

log = logging.getLogger(__name__)

try:
    from transformers import AutoConfig, WhisperForConditionalGeneration, WhisperProcessor

    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

DEFAULT_HF_FINETUNED_REPO = "Ignites/fine_tuned_med_whisper_rus"
DEFAULT_ASR_CHUNK_SECONDS = 30.0
WHISPER_DYNAMIC_INITIAL_PROMPT_MAX_CHARS = 2400

_clinvoice_model_load_lock = threading.Lock()
_clinvoice_fw_whisper_models: dict[tuple[str, str, str], object] = {}
_clinvoice_openai_whisper_models: dict[tuple[str, str], object] = {}
_transformers_bundles: dict[str, dict] = {}


def _e(name: str) -> str:
    return (os.environ.get(name) or "").strip()


def resolve_asr_chunk_seconds() -> float:
    raw = _e("CLINVOICE_ASR_CHUNK_SECONDS")
    if raw:
        try:
            v = float(raw)
            if v > 0:
                return v
        except ValueError:
            pass
    return DEFAULT_ASR_CHUNK_SECONDS


def resolve_hub_model_id() -> str:
    return _e("CLINVOICE_HF_MODEL_REPO") or DEFAULT_HF_FINETUNED_REPO


def resolve_whisper_engine() -> str:
    raw = _e("CLINVOICE_WHISPER_ENGINE").lower()
    if raw in ("transformers", "pytorch", "hf", "huggingface"):
        return "transformers"
    if raw in ("faster_whisper", "faster-whisper", "ct2", "ctranslate2"):
        return "faster_whisper"
    return "faster_whisper"


def resolve_draft_beam_size() -> int:
    raw = _e("CLINVOICE_DRAFT_BEAM_SIZE")
    if raw.isdigit():
        return max(1, min(5, int(raw)))
    return 1


def resolve_draft_vad_filter() -> bool:
    raw = _e("CLINVOICE_DRAFT_VAD_FILTER").lower()
    if raw in ("0", "false", "no", "off"):
        return False
    if raw in ("1", "true", "yes", "on"):
        return True
    return True


def _resolve_faster_whisper_no_speech_threshold(*, draft: bool) -> float:
    raw = _e("CLINVOICE_WHISPER_NO_SPEECH_THRESHOLD")
    if draft:
        dr = _e("CLINVOICE_DRAFT_NO_SPEECH_THRESHOLD")
        if dr:
            raw = dr
    else:
        fn = _e("CLINVOICE_FINAL_NO_SPEECH_THRESHOLD")
        if fn:
            raw = fn
    default = 0.66 if draft else 0.62
    if not raw:
        v = default
    else:
        try:
            v = float(raw)
        except ValueError:
            v = default
    return max(0.35, min(0.95, v))


def _resolve_faster_whisper_compression_ratio(*, draft: bool) -> float:
    raw = _e("CLINVOICE_WHISPER_COMPRESSION_RATIO_THRESHOLD")
    if draft:
        dr = _e("CLINVOICE_DRAFT_COMPRESSION_RATIO_THRESHOLD")
        if dr:
            raw = dr
    else:
        fn = _e("CLINVOICE_FINAL_COMPRESSION_RATIO_THRESHOLD")
        if fn:
            raw = fn
    default = 2.0 if draft else 2.35
    if not raw:
        v = default
    else:
        try:
            v = float(raw)
        except ValueError:
            v = default
    return max(1.2, min(3.5, v))


def _resolve_whisper_initial_prompt() -> str:
    raw = _e("CLINVOICE_WHISPER_INITIAL_PROMPT")
    return (raw[:224] if raw else "").strip()


def openai_whisper_download_dir() -> str:
    return os.environ.get("_CLINVOICE_OPENAI_WHISPER_DIR") or os.path.join(
        resolve_app_cache_root(), "openai-whisper"
    )


def hf_hub_download_dir() -> str:
    return os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.join(resolve_app_cache_root(), "huggingface")),
        "hub",
    )


def infer_whisper_processor_repo(hub_model_id: str) -> str:
    explicit = _e("CLINVOICE_WHISPER_BASE_REPO")
    if explicit:
        return explicit
    try:
        cfg = AutoConfig.from_pretrained(hub_model_id)
        cand = getattr(cfg, "_name_or_path", None) or getattr(cfg, "name_or_path", None)
        if cand:
            s = str(cand).strip()
            if s.startswith("openai/whisper-"):
                return s
    except Exception:
        pass
    return "openai/whisper-small"


def _load_faster_whisper_cached(repo_id: str, device_kw: str, compute_type: str):
    from faster_whisper import WhisperModel

    k = (str(repo_id), str(device_kw), str(compute_type))
    with _clinvoice_model_load_lock:
        if k not in _clinvoice_fw_whisper_models:
            _clinvoice_fw_whisper_models[k] = WhisperModel(
                repo_id,
                device=device_kw,
                compute_type=compute_type,
                download_root=hf_hub_download_dir(),
            )
        return _clinvoice_fw_whisper_models[k]


def _load_openai_whisper_cached(model_size: str):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    k = (str(model_size), dev)
    with _clinvoice_model_load_lock:
        if k not in _clinvoice_openai_whisper_models:
            _clinvoice_openai_whisper_models[k] = whisper.load_model(
                model_size, device=dev, download_root=openai_whisper_download_dir()
            )
        return _clinvoice_openai_whisper_models[k]


def transcribe_wav_in_chunks(
    transcriber: "AudioTranscriberWithMetrics",
    wav_path: str,
    language: str = "ru",
    *,
    draft: bool = False,
    initial_prompt: Optional[str] = None,
) -> str:
    chunk_sec = resolve_asr_chunk_seconds()
    with wave.open(wav_path, "rb") as w:
        ch = w.getnchannels()
        sw = w.getsampwidth()
        fr = w.getframerate()
        nframes = w.getnframes()
        if ch != 1:
            raise ValueError("Ожидается моно WAV")
        frames_bytes = w.readframes(nframes)
    duration = nframes / float(fr)
    if duration <= chunk_sec:
        return transcriber.transcribe_audio(
            wav_path, language=language, draft=draft, initial_prompt=initial_prompt
        )

    chunk_frames = max(1, int(fr * chunk_sec))
    parts: List[str] = []
    offset = 0
    while offset < nframes:
        take = min(chunk_frames, nframes - offset)
        start_byte = offset * ch * sw
        end_byte = (offset + take) * ch * sw
        chunk_data = frames_bytes[start_byte:end_byte]
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".wav")
            os.close(fd)
            with wave.open(tmp_path, "wb") as cw:
                cw.setnchannels(ch)
                cw.setsampwidth(sw)
                cw.setframerate(fr)
                cw.writeframes(chunk_data)
            parts.append(
                transcriber.transcribe_audio(
                    tmp_path, language=language, draft=draft, initial_prompt=initial_prompt
                )
            )
        finally:
            if tmp_path and os.path.isfile(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
        offset += take
    return " ".join(p.strip() for p in parts if p.strip()).strip()


class AudioTranscriberWithMetrics:
    def __init__(
        self,
        model_size: str = "base",
        hub_model_id: Optional[str] = None,
        *,
        silent_ui: bool = False,
    ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        engine = resolve_whisper_engine() if hub_model_id else "openai"
        safe_hub = (hub_model_id or "").replace("/", "_")
        transformers_cache_key = f"asr_transformers_{model_size}_{safe_hub or 'openai_whisper'}"
        self.use_faster_whisper = False
        self.use_transformers = False

        if hub_model_id and engine == "faster_whisper":
            if not silent_ui:
                log.info("Загрузка faster-whisper с Hub: %s", hub_model_id)
            device_kw = "cuda" if self.device.type == "cuda" else "cpu"
            compute_type = "float16" if device_kw == "cuda" else "int8"
            try:
                self.faster_model = _load_faster_whisper_cached(
                    hub_model_id, device_kw, compute_type
                )
            except ImportError as e:
                raise RuntimeError("Нужен пакет faster-whisper") from e
            except Exception as e:
                raise RuntimeError(f"Ошибка загрузки модели с Hub: {e}") from e
            self.use_faster_whisper = True
            self.use_transformers = False
            self._draft_beam_size = resolve_draft_beam_size()
            self._draft_vad_filter = resolve_draft_vad_filter()
            self._fw_ns_draft = _resolve_faster_whisper_no_speech_threshold(draft=True)
            self._fw_ns_final = _resolve_faster_whisper_no_speech_threshold(draft=False)
            self._fw_cr_draft = _resolve_faster_whisper_compression_ratio(draft=True)
            self._fw_cr_final = _resolve_faster_whisper_compression_ratio(draft=False)
            self._fw_initial_prompt = _resolve_whisper_initial_prompt()
        elif hub_model_id:
            if not TRANSFORMERS_AVAILABLE:
                raise RuntimeError("Пакет transformers необходим для дообученной модели (PyTorch).")
            if transformers_cache_key not in _transformers_bundles:
                if not silent_ui:
                    log.info("Загрузка PyTorch Whisper с Hub: %s", hub_model_id)
                try:
                    model = WhisperForConditionalGeneration.from_pretrained(
                        hub_model_id,
                        torch_dtype=torch.float32,
                    )
                    try:
                        processor = WhisperProcessor.from_pretrained(hub_model_id)
                    except Exception:
                        base_proc = infer_whisper_processor_repo(hub_model_id)
                        if not silent_ui:
                            log.warning(
                                "Процессор из %s не найден в %s, загружаю с %s",
                                hub_model_id,
                                hub_model_id,
                                base_proc,
                            )
                        processor = WhisperProcessor.from_pretrained(base_proc)
                    _transformers_bundles[transformers_cache_key] = {
                        "model": model,
                        "processor": processor,
                        "feature_extractor": processor.feature_extractor,
                        "tokenizer": processor.tokenizer,
                    }
                except Exception as e:
                    raise RuntimeError(f"Ошибка загрузки модели с Hub: {e}") from e
            cached = _transformers_bundles[transformers_cache_key]
            self.model = cached["model"]
            self.processor = cached["processor"]
            self.feature_extractor = cached["feature_extractor"]
            self.tokenizer = cached["tokenizer"]
            self.use_transformers = True
            self.use_faster_whisper = False
            self.model.to(self.device)
            self.model.eval()
            self._draft_beam_size = resolve_draft_beam_size()
            self._draft_vad_filter = resolve_draft_vad_filter()
            self._fw_ns_draft = 0.6
            self._fw_ns_final = 0.6
            self._fw_cr_draft = 2.4
            self._fw_cr_final = 2.4
            self._fw_initial_prompt = _resolve_whisper_initial_prompt()
        else:
            if not silent_ui:
                log.info("Загрузка openai-whisper: %s", model_size)
            self.model = _load_openai_whisper_cached(model_size)
            self.use_transformers = False
            self.use_faster_whisper = False
            self._draft_beam_size = 1
            self._draft_vad_filter = False
            self._fw_ns_draft = 0.6
            self._fw_ns_final = 0.6
            self._fw_cr_draft = 2.4
            self._fw_cr_final = 2.4
            self._fw_initial_prompt = ""

    def transcribe_audio(
        self,
        audio_path,
        language: str = "ru",
        *,
        draft: bool = False,
        initial_prompt: Optional[str] = None,
    ):
        if getattr(self, "use_faster_whisper", False):
            beam = getattr(self, "_draft_beam_size", 1) if draft else 5
            vad = getattr(self, "_draft_vad_filter", True) if draft else False
            ns = getattr(self, "_fw_ns_draft" if draft else "_fw_ns_final", 0.62)
            cr = getattr(self, "_fw_cr_draft" if draft else "_fw_cr_final", 2.35)
            if initial_prompt is not None:
                ip = (initial_prompt or "").strip()[:WHISPER_DYNAMIC_INITIAL_PROMPT_MAX_CHARS]
                if not ip:
                    ip = (getattr(self, "_fw_initial_prompt", None) or "").strip()
            else:
                ip = (getattr(self, "_fw_initial_prompt", None) or "").strip()
            kw = dict(
                language=language,
                beam_size=beam,
                vad_filter=vad,
                no_speech_threshold=ns,
                compression_ratio_threshold=cr,
                condition_on_previous_text=False if draft else True,
            )
            if ip:
                kw["initial_prompt"] = ip
            segments, _info = self.faster_model.transcribe(audio_path, **kw)
            return "".join(seg.text for seg in segments).strip()

        if self.use_transformers:
            import librosa

            audio, _sr = librosa.load(audio_path, sr=16000)
            dev = next(self.model.parameters()).device
            inputs = self.feature_extractor(
                audio, sampling_rate=16000, return_tensors="pt"
            )
            input_features = inputs["input_features"].to(dev)
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(
                language=language, task="transcribe"
            )
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    input_features, forced_decoder_ids=forced_decoder_ids
                )
            transcription = self.processor.batch_decode(
                predicted_ids, skip_special_tokens=True
            )[0]
            return transcription

        ip = None
        if initial_prompt is not None:
            t = (initial_prompt or "").strip()[:WHISPER_DYNAMIC_INITIAL_PROMPT_MAX_CHARS]
            ip = t if t else None
        if ip is None:
            t2 = (getattr(self, "_fw_initial_prompt", None) or "").strip()
            ip = t2 if t2 else None
        if ip:
            result = self.model.transcribe(audio_path, language=language, initial_prompt=ip)
        else:
            result = self.model.transcribe(audio_path, language=language)
        return result["text"]


def pcm_bytes_to_transcribe_path(pcm: bytes) -> str:
    """Временный WAV mono 16k из PCM (вызывающий удаляет файл)."""
    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    with open(path, "wb") as wf:
        wf.write(pcm_mono_s16le_to_wav_bytes(pcm))
    return path


__all__ = [
    "AudioTranscriberWithMetrics",
    "DEFAULT_ASR_CHUNK_SECONDS",
    "WHISPER_DYNAMIC_INITIAL_PROMPT_MAX_CHARS",
    "hf_hub_download_dir",
    "infer_whisper_processor_repo",
    "openai_whisper_download_dir",
    "pcm_bytes_to_transcribe_path",
    "resolve_asr_chunk_seconds",
    "resolve_hub_model_id",
    "resolve_whisper_engine",
    "transcribe_wav_in_chunks",
]
