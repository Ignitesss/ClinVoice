# -*- coding: utf-8 -*-
"""
ClinVoice - Medical Speech Recognition App (врачебный сценарий: запись, транскрибация, протокол).
"""

import os

import streamlit as st
import streamlit.components.v1 as components


def resolve_app_cache_root() -> str:
    """Корень кэша артефактов: env CLINVOICE_CACHE_DIR, секрет Streamlit, или ~/.cache/clinvoice."""
    root = (os.environ.get("CLINVOICE_CACHE_DIR") or "").strip()
    if not root:
        try:
            if hasattr(st, "secrets") and st.secrets and "CLINVOICE_CACHE_DIR" in st.secrets:
                root = str(st.secrets["CLINVOICE_CACHE_DIR"]).strip()
        except Exception:
            pass
    if not root:
        root = os.path.join(os.path.expanduser("~"), ".cache", "clinvoice")
    os.makedirs(root, exist_ok=True)
    return root


def apply_disk_cache_layout(cache_root: str) -> None:
    """
    Свести загрузки к одному дереву (меньше разброса по ~/.cache).
    Hugging Face: HF_HOME / HF_HUB_CACHE; openai-whisper — отдельная подпапка (load_model).
    """
    hf_default = os.path.join(cache_root, "huggingface")
    openai_dir = os.path.join(cache_root, "openai-whisper")
    torch_dir = os.path.join(cache_root, "torch")
    for d in (hf_default, openai_dir, torch_dir):
        os.makedirs(d, exist_ok=True)
    os.environ.setdefault("HF_HOME", hf_default)
    hf_home = os.environ["HF_HOME"]
    os.makedirs(hf_home, exist_ok=True)
    hub = os.path.join(hf_home, "hub")
    os.environ.setdefault("HF_HUB_CACHE", hub)
    os.makedirs(os.environ["HF_HUB_CACHE"], exist_ok=True)
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_home, "transformers"))
    os.makedirs(os.environ["TRANSFORMERS_CACHE"], exist_ok=True)
    os.environ.setdefault("TORCH_HOME", torch_dir)
    os.environ["_CLINVOICE_OPENAI_WHISPER_DIR"] = openai_dir


_APP_CACHE_ROOT = resolve_app_cache_root()
apply_disk_cache_layout(_APP_CACHE_ROOT)

import base64
import io
import json
import tempfile
import threading
import wave
import whisper
from docx import Document
import torch
from datetime import datetime, timedelta
from typing import List, Optional
from zoneinfo import ZoneInfo

from streamlit_webrtc import RTCConfiguration, WebRtcMode, webrtc_streamer

from protocol import (
    PROTOCOL_FIELD_KEYS,
    fill_protocol_from_transcript,
    format_protocol_editor_text,
    parse_protocol_editor_text,
    resolve_yandex_api_key,
    resolve_yandex_folder_id,
    resolve_yandex_iam_token,
    yandex_llm_configured,
)
from speechkit_stream import speechkit_stt_configured
from webrtc_draft import DraftAudioProcessor

# For loading fine-tuned Whisper models
try:
    from transformers import AutoConfig, WhisperForConditionalGeneration, WhisperProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

DEFAULT_HF_FINETUNED_REPO = "Ignites/fine_tuned_med_whisper_rus"
DEFAULT_ASR_CHUNK_SECONDS = 30.0

SPEECHKIT_RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)


def speechkit_processor_factory():
    return DraftAudioProcessor(
        st.session_state.speechkit_shared,
        resolve_yandex_api_key(),
        resolve_yandex_folder_id(),
        resolve_yandex_iam_token(),
    )


def resolve_asr_chunk_seconds() -> float:
    raw = (os.environ.get("CLINVOICE_ASR_CHUNK_SECONDS") or "").strip()
    if raw:
        try:
            v = float(raw)
            if v > 0:
                return v
        except ValueError:
            pass
    try:
        if hasattr(st, "secrets") and st.secrets and "CLINVOICE_ASR_CHUNK_SECONDS" in st.secrets:
            v = float(str(st.secrets["CLINVOICE_ASR_CHUNK_SECONDS"]).strip())
            if v > 0:
                return v
    except Exception:
        pass
    return DEFAULT_ASR_CHUNK_SECONDS


def _read_wav_params(data: bytes):
    with wave.open(io.BytesIO(data), "rb") as w:
        nframes = w.getnframes()
        return w.getnchannels(), w.getsampwidth(), w.getframerate(), nframes, w.readframes(nframes)


def merge_wav_segments_to_bytes(segments: List[bytes]) -> bytes:
    """Склеить несколько WAV с одинаковыми параметрами (mono, один sampwidth, одна частота)."""
    if not segments:
        raise ValueError("Нет сегментов для склейки")
    first_ch, first_sw, first_fr, _first_nf, first_frames = _read_wav_params(segments[0])
    if first_ch != 1:
        raise ValueError("Ожидается моно WAV (1 канал)")
    if first_fr != 16000:
        raise ValueError(f"Ожидается частота 16000 Hz, получено {first_fr}")
    all_frames = [first_frames]
    for i, seg in enumerate(segments[1:], start=2):
        ch, sw, fr, _nf, frames = _read_wav_params(seg)
        if (ch, sw, fr) != (first_ch, first_sw, first_fr):
            raise ValueError(f"Сегмент {i}: несовпадение формата WAV с первым фрагментом")
        all_frames.append(frames)
    out = io.BytesIO()
    with wave.open(out, "wb") as wout:
        wout.setnchannels(first_ch)
        wout.setsampwidth(first_sw)
        wout.setframerate(first_fr)
        for frame_chunk in all_frames:
            wout.writeframes(frame_chunk)
    return out.getvalue()


def merge_wav_segments_to_file(segments: List[bytes], out_path: str) -> None:
    data = merge_wav_segments_to_bytes(segments)
    with open(out_path, "wb") as f:
        f.write(data)


def transcribe_wav_in_chunks(transcriber: "AudioTranscriberWithMetrics", wav_path: str, language: str = "ru") -> str:
    """Длинный WAV режется на части ≤ resolve_asr_chunk_seconds(); короткий — один вызов transcribe_audio."""
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
        return transcriber.transcribe_audio(wav_path, language=language)

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
            parts.append(transcriber.transcribe_audio(tmp_path, language=language))
        finally:
            if tmp_path and os.path.isfile(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
        offset += take
    return " ".join(p.strip() for p in parts if p.strip()).strip()


def resolve_hub_model_id() -> str:
    env_id = (os.environ.get("CLINVOICE_HF_MODEL_REPO") or "").strip()
    if env_id:
        return env_id
    try:
        if hasattr(st, "secrets") and "CLINVOICE_HF_MODEL_REPO" in st.secrets:
            return str(st.secrets["CLINVOICE_HF_MODEL_REPO"]).strip()
    except Exception:
        pass
    return DEFAULT_HF_FINETUNED_REPO


def resolve_whisper_engine() -> str:
    """
    Репозиторий на HF: «faster_whisper» (CTranslate2: model.bin) или «transformers»
    (PyTorch: model.safetensors / pytorch_model.bin). Задаётся CLINVOICE_WHISPER_ENGINE
    в env или Streamlit Secrets. По умолчанию — faster_whisper (репозиторий по умолчанию
    в формате CT2). Для старых репо только с PyTorch укажите: transformers | pytorch | hf.
    """
    raw = (os.environ.get("CLINVOICE_WHISPER_ENGINE") or "").strip().lower()
    if raw in ("transformers", "pytorch", "hf", "huggingface"):
        return "transformers"
    if raw in ("faster_whisper", "faster-whisper", "ct2", "ctranslate2"):
        return "faster_whisper"
    try:
        if hasattr(st, "secrets") and "CLINVOICE_WHISPER_ENGINE" in st.secrets:
            v = str(st.secrets["CLINVOICE_WHISPER_ENGINE"]).strip().lower()
            if v in ("transformers", "pytorch", "hf", "huggingface"):
                return "transformers"
            if v in ("faster_whisper", "faster-whisper", "ct2", "ctranslate2"):
                return "faster_whisper"
    except Exception:
        pass
    return "faster_whisper"


def infer_whisper_processor_repo(hub_model_id: str) -> str:
    """Базовый openai/whisper-* для процессора, если в finetune-репо нет preprocessor_config.json / tokenizer."""
    explicit = (os.environ.get("CLINVOICE_WHISPER_BASE_REPO") or "").strip()
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


def openai_whisper_download_dir() -> str:
    return os.environ.get("_CLINVOICE_OPENAI_WHISPER_DIR") or os.path.join(
        _APP_CACHE_ROOT, "openai-whisper"
    )


def hf_hub_download_dir() -> str:
    return os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME", os.path.join(_APP_CACHE_ROOT, "huggingface")),
        "hub",
    )


@st.cache_resource(show_spinner=False)
def _load_faster_whisper_cached(repo_id: str, device_kw: str, compute_type: str):
    from faster_whisper import WhisperModel

    return WhisperModel(
        repo_id,
        device=device_kw,
        compute_type=compute_type,
        download_root=hf_hub_download_dir(),
    )


@st.cache_resource(show_spinner=False)
def _load_openai_whisper_cached(model_size: str):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    return whisper.load_model(
        model_size, device=dev, download_root=openai_whisper_download_dir()
    )


# Page config
st.set_page_config(
    page_title="ClinVoice",
    page_icon="🏥",
    layout="wide"
)

# ============ CLASSES FROM YOUR COLAB ============


class AudioTranscriberWithMetrics:
    def __init__(
        self,
        model_size="base",
        hub_model_id: Optional[str] = None,
        *,
        silent_ui: bool = False,
    ):
        """Whisper: openai-whisper, HF (PyTorch/transformers) или HF (CTranslate2 / faster-whisper)."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        engine = resolve_whisper_engine() if hub_model_id else "openai"
        safe_hub = (hub_model_id or "").replace("/", "_")
        transformers_cache_key = f"asr_transformers_{model_size}_{safe_hub or 'openai_whisper'}"
        self.use_faster_whisper = False
        self.use_transformers = False

        if hub_model_id and engine == "faster_whisper":
            if not silent_ui:
                st.info(
                    f"Загрузка модели (формат CTranslate2) с Hugging Face: {hub_model_id} "
                    f"(первый запуск может занять несколько минут)..."
                )
            device_kw = "cuda" if self.device.type == "cuda" else "cpu"
            compute_type = "float16" if device_kw == "cuda" else "int8"
            try:
                self.faster_model = _load_faster_whisper_cached(
                    hub_model_id, device_kw, compute_type
                )
            except ImportError:
                st.error("Нужен пакет faster-whisper. Выполните: pip install -r requirements.txt")
                st.stop()
            except Exception as e:
                st.error(f"Ошибка загрузки модели с Hub: {e}")
                st.stop()
            self.use_faster_whisper = True
            self.use_transformers = False
        elif hub_model_id:
            if not TRANSFORMERS_AVAILABLE:
                st.error("Пакет transformers необходим для дообученной модели с Hub (режим PyTorch).")
                st.stop()
            if transformers_cache_key not in st.session_state:
                if not silent_ui:
                    st.info(
                        f"Загрузка дообученной модели (PyTorch) с Hugging Face: {hub_model_id} "
                        f"(первый запуск может занять несколько минут)..."
                    )
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
                            st.warning(
                                f"В репозитории «{hub_model_id}» нет полного процессора "
                                f"(нужен preprocessor_config.json и файлы токенизатора). "
                                f"Загружаю процессор с **{base_proc}**. "
                                f"Если размер не совпадает с дообучением, задайте переменную **CLINVOICE_WHISPER_BASE_REPO** "
                                f"(например, `openai/whisper-base`)."
                            )
                        processor = WhisperProcessor.from_pretrained(base_proc)
                    st.session_state[transformers_cache_key] = {
                        "model": model,
                        "processor": processor,
                        "feature_extractor": processor.feature_extractor,
                        "tokenizer": processor.tokenizer,
                    }
                except Exception as e:
                    st.error(f"Ошибка загрузки модели с Hub: {e}")
                    st.stop()
            cached = st.session_state[transformers_cache_key]
            self.model = cached["model"]
            self.processor = cached["processor"]
            self.feature_extractor = cached["feature_extractor"]
            self.tokenizer = cached["tokenizer"]
            self.use_transformers = True
            self.use_faster_whisper = False
            self.model.to(self.device)
            self.model.eval()
        else:
            if not silent_ui:
                st.info(f"Загрузка базовой модели Whisper ({model_size})...")
            self.model = _load_openai_whisper_cached(model_size)
            self.use_transformers = False
            self.use_faster_whisper = False

    def transcribe_audio(self, audio_path, language='ru'):
        """Транскрибация аудиофайла"""
        if getattr(self, "use_faster_whisper", False):
            segments, _info = self.faster_model.transcribe(
                audio_path,
                language=language,
                beam_size=5,
                vad_filter=False,
            )
            return "".join(seg.text for seg in segments).strip()

        if self.use_transformers:
            # Use Transformers for fine-tuned model
            import librosa
            audio, sr = librosa.load(audio_path, sr=16000)
            dev = next(self.model.parameters()).device
            inputs = self.feature_extractor(
                audio, sampling_rate=16000, return_tensors="pt"
            )
            input_features = inputs["input_features"].to(dev)
            forced_decoder_ids = self.processor.get_decoder_prompt_ids(language=language, task="transcribe")
            with torch.no_grad():
                predicted_ids = self.model.generate(
                    input_features, forced_decoder_ids=forced_decoder_ids
                )
            transcription = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
            return transcription
        else:
            # Use openai-whisper
            result = self.model.transcribe(audio_path, language=language)
            return result["text"]


# ============ DOCX / TXT GENERATION ============

def format_consultation_date_gmt3() -> str:
    """Дата/время в поясе Europe/Moscow (UTC+3)."""
    return datetime.now(ZoneInfo("Europe/Moscow")).strftime("%d.%m.%Y %H:%M (GMT+3)")


def create_structured_protocol_docx(fields: dict, consultation_date: str):
    """Протокол: дата → жалобы → анамнез → заключение → рекомендации."""
    doc = Document()
    doc.add_heading("Протокол консультации", 0)
    doc.add_paragraph(f"Дата: {consultation_date}")
    sections = [
        ("Жалобы", fields.get("complaints", "")),
        ("Анамнез", fields.get("anamnesis", "")),
        ("Заключение", fields.get("conclusion", "")),
        ("Рекомендации", fields.get("recommendations", "")),
    ]
    for title, body in sections:
        doc.add_heading(title, level=1)
        doc.add_paragraph(body if (body or "").strip() else "—")
    return doc


def trigger_browser_text_download(filename: str, text: str) -> None:
    """Один раз за отрисовку: инициировать скачивание текстового файла в браузере (data URL)."""
    b64 = base64.b64encode(text.encode("utf-8")).decode("ascii")
    fname_js = json.dumps(filename)
    components.html(
        f"""
        <script>
            const link = document.createElement("a");
            link.setAttribute("href", "data:text/plain;charset=utf-8;base64,{b64}");
            link.setAttribute("download", {fname_js});
            document.body.appendChild(link);
            link.click();
            link.remove();
        </script>
        """,
        height=0,
        width=0,
    )


def build_structured_protocol_txt(fields: dict, consultation_date: str) -> str:
    parts = [
        "Протокол консультации",
        "",
        f"Дата: {consultation_date}",
        "",
        "Жалобы",
        fields.get("complaints", "") or "—",
        "",
        "Анамнез",
        fields.get("anamnesis", "") or "—",
        "",
        "Заключение",
        fields.get("conclusion", "") or "—",
        "",
        "Рекомендации",
        fields.get("recommendations", "") or "—",
        "",
    ]
    return "\n".join(parts)


# ============ MAIN UI ============

st.title("🏥 ClinVoice")
st.markdown("**Распознавание речи для медицинских консультаций**")

hub_model_id = resolve_hub_model_id()
model_size = "small"

st.header("Запись консультации")
if "original_transcription" not in st.session_state:
    st.session_state.original_transcription = None
if "protocol_editor_text" not in st.session_state:
    st.session_state.protocol_editor_text = ""
if "doctor_audio_segments" not in st.session_state:
    st.session_state.doctor_audio_segments = []
if "doctor_recorder_nonce" not in st.session_state:
    st.session_state.doctor_recorder_nonce = 0
if "speechkit_shared" not in st.session_state:
    st.session_state.speechkit_shared = {
        "draft": "",
        "error": None,
        "lock": threading.Lock(),
    }
st.warning(
    "При первом запуске загрузка и настройка приложения могут занять несколько минут — это нормально."
)
st.caption(
    "Записывайте фрагменты по очереди: старт/стоп микрофона, затем «Добавить фрагмент». "
    "Так можно делать паузы между частями консультации. Длинное аудио распознаётся по частям "
    f"до {int(resolve_asr_chunk_seconds())} с — меньше нагрузка на память."
)

if speechkit_stt_configured(resolve_yandex_api_key(), resolve_yandex_iam_token()):
    st.subheader("Черновик во время речи (SpeechKit)")
    st.caption(
        "Ниже — потоковое распознавание Yandex SpeechKit для подсказки во время диктовки. "
        "Итоговый транскрипт и протокол формируются **только из Whisper** после кнопки транскрибации. "
        "В каталоге Yandex Cloud сервисному аккаунту нужна роль **ai.speechkit-stt.user**; SpeechKit тарифицируется отдельно от YandexGPT."
    )
    webrtc_streamer(
        key="clinvoice_speechkit_mic",
        mode=WebRtcMode.SENDONLY,
        rtc_configuration=SPEECHKIT_RTC_CONFIGURATION,
        media_stream_constraints={"audio": True, "video": False},
        audio_processor_factory=speechkit_processor_factory,
        async_processing=True,
        sendback_audio=False,
    )

    @st.fragment(run_every=timedelta(milliseconds=450))
    def _speechkit_draft_fragment():
        sh = st.session_state.speechkit_shared
        lk = sh.get("lock")
        draft = ""
        err: Optional[str] = None
        if lk:
            with lk:
                draft = sh.get("draft") or ""
                err = sh.get("error")
        if err:
            st.error(err)
        elif draft.strip():
            st.info(draft)
        else:
            st.caption("Включите запись в блоке выше — здесь появится черновик.")

    _speechkit_draft_fragment()
    if st.button("Сбросить черновик SpeechKit", key="speechkit_reset_draft"):
        lk = st.session_state.speechkit_shared.get("lock")
        if lk:
            with lk:
                st.session_state.speechkit_shared["draft"] = ""
                st.session_state.speechkit_shared["error"] = None
        st.rerun()
else:
    st.info(
        "Черновик SpeechKit недоступен: задайте **YANDEX_CLOUD_API_KEY** или **YANDEX_IAM_TOKEN** "
        "(как для протокола). Опционально **YANDEX_FOLDER_ID** — передаётся в запрос SpeechKit."
    )

seg_count = len(st.session_state.doctor_audio_segments)
st.caption(f"Сохранено фрагментов в консультации: **{seg_count}**")

rec_key = f"doctor_recorder_{st.session_state.doctor_recorder_nonce}"
audio_file = st.audio_input("🎙️ Запись фрагмента", key=rec_key, sample_rate=16000)

c1, c2, c3 = st.columns(3)
with c1:
    if st.button("Добавить фрагмент в консультацию"):
        if audio_file is None:
            st.warning("Сначала запишите фрагмент микрофоном.")
        else:
            st.session_state.doctor_audio_segments.append(bytes(audio_file.getbuffer()))
            st.session_state.doctor_recorder_nonce += 1
            st.rerun()
with c2:
    if st.button("Удалить последний фрагмент"):
        if st.session_state.doctor_audio_segments:
            st.session_state.doctor_audio_segments.pop()
            st.rerun()
        else:
            st.info("Нет сохранённых фрагментов.")
with c3:
    if st.button("Очистить все фрагменты"):
        st.session_state.doctor_audio_segments = []
        st.session_state.doctor_recorder_nonce += 1
        st.rerun()

if st.button("Транскрибировать и заполнить протокол", type="primary"):
    if not yandex_llm_configured():
        st.error(
            "Не заданы параметры для заполнения протокола: укажите "
            "**YANDEX_FOLDER_ID** и (**YANDEX_CLOUD_API_KEY** или **YANDEX_IAM_TOKEN**) "
            "в переменных окружения или в секретах приложения."
        )
        st.stop()

    to_merge = list(st.session_state.doctor_audio_segments)
    if audio_file is not None:
        buf = bytes(audio_file.getbuffer())
        if not to_merge or to_merge[-1] != buf:
            to_merge.append(buf)

    if not to_merge:
        st.error("Добавьте хотя бы один аудиофрагмент или запишите и оставьте запись в микрофоне.")
        st.stop()

    merged_path = "/tmp/doctor_merged_consultation.wav"
    try:
        merge_wav_segments_to_file(to_merge, merged_path)
    except ValueError as e:
        st.error(str(e))
        st.stop()

    try:
        with st.spinner("Транскрибация... Это может занять несколько минут."):
            transcriber = AudioTranscriberWithMetrics(
                model_size=model_size, hub_model_id=hub_model_id, silent_ui=True
            )
            transcription = transcribe_wav_in_chunks(transcriber, merged_path, language="ru")
    except Exception as e:
        st.error(f"Ошибка транскрибации: {e}")
        st.stop()
    finally:
        if os.path.isfile(merged_path):
            try:
                os.remove(merged_path)
            except OSError:
                pass

    st.session_state.doctor_audio_segments = []
    st.session_state.doctor_recorder_nonce += 1
    st.session_state.original_transcription = transcription
    st.session_state.doctor_transcript_editor = transcription
    st.session_state.protocol_consultation_date = format_consultation_date_gmt3()

    try:
        with st.spinner("Заполнение протокола..."):
            protocol = fill_protocol_from_transcript(transcription)
        st.session_state.protocol_editor_text = format_protocol_editor_text(
            st.session_state.protocol_consultation_date,
            protocol,
        )
        st.success(
            "Готово. Проверьте и при необходимости отредактируйте протокол ниже. "
            "Файл **protocol.txt** также должен начать скачиваться автоматически; "
            "при блокировке браузером сохраните протокол кнопкой ниже (.txt или .docx)."
        )
        _auto_txt = build_structured_protocol_txt(protocol, st.session_state.protocol_consultation_date)
        trigger_browser_text_download("protocol.txt", _auto_txt)
    except Exception as e:
        st.error(f"Ошибка заполнения протокола: {e}")
        st.session_state.protocol_editor_text = format_protocol_editor_text(
            st.session_state.protocol_consultation_date,
            {k: "" for k in PROTOCOL_FIELD_KEYS},
        )

if st.session_state.original_transcription:
    if "doctor_transcript_editor" not in st.session_state:
        st.session_state.doctor_transcript_editor = st.session_state.original_transcription

    st.header("Протокол консультации")

    with st.expander("Транскрипт (справочно, можно исправить ошибки распознавания)"):
        st.text_area(
            "Транскрипт",
            height=180,
            key="doctor_transcript_editor",
            label_visibility="collapsed",
        )

    st.text_area(
        "Протокол (редактирование)",
        height=320,
        key="protocol_editor_text",
        help="Формат: строки «Дата:», «Жалобы:», «Анамнез:», «Заключение:», «Рекомендации:» с двоеточием.",
    )

    _parsed_date, fields = parse_protocol_editor_text(
        st.session_state.get("protocol_editor_text", "")
    )
    consultation_date = (
        _parsed_date.strip()
        or st.session_state.get("protocol_consultation_date")
        or format_consultation_date_gmt3()
    )

    st.subheader("Скачать протокол")
    doc = create_structured_protocol_docx(fields, consultation_date)
    doc_buf = io.BytesIO()
    doc.save(doc_buf)
    doc_bytes = doc_buf.getvalue()
    txt_body = build_structured_protocol_txt(fields, consultation_date)

    col_docx, col_txt = st.columns(2)
    col_docx.download_button(
        "📄 protocol.docx",
        doc_bytes,
        "protocol.docx",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    )
    col_txt.download_button(
        "📄 protocol.txt",
        txt_body,
        "protocol.txt",
        "text/plain",
    )

# Footer
st.markdown("---")
st.markdown("*ClinVoice v0.4 - Medical ASR Tool*")
