# -*- coding: utf-8 -*-
"""
ClinVoice - Medical Speech Recognition App
Full version with AudioTranscriber and WhisperFineTuner
"""

import os
import streamlit as st


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

import whisper
import jiwer
from rouge import Rouge
from docx import Document
import torch
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from protocol import (
    PROTOCOL_FIELD_KEYS,
    fill_protocol_from_transcript,
    format_protocol_editor_text,
    parse_protocol_editor_text,
    yandex_llm_configured,
)

# For loading fine-tuned Whisper models
try:
    from transformers import AutoConfig, WhisperForConditionalGeneration, WhisperProcessor
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

DEFAULT_HF_FINETUNED_REPO = "Ignites/fine_tuned_med_whisper_rus"


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


def get_session_rouge() -> Rouge:
    if "_clinvoice_rouge" not in st.session_state:
        st.session_state["_clinvoice_rouge"] = Rouge()
    return st.session_state["_clinvoice_rouge"]


_WER_TRANSFORMATION = jiwer.Compose(
    [
        jiwer.ToLowerCase(),
        jiwer.RemoveWhiteSpace(replace_by_space=True),
        jiwer.RemoveMultipleSpaces(),
        jiwer.Strip(),
        jiwer.RemovePunctuation(),
        jiwer.ReduceToListOfListOfWords(),
    ]
)


def compute_wer_metrics(reference: str, hypothesis: str) -> dict:
    """WER и связанные счётчики (без загрузки ASR-модели)."""
    wer_score = jiwer.wer(
        reference,
        hypothesis,
        reference_transform=_WER_TRANSFORMATION,
        hypothesis_transform=_WER_TRANSFORMATION,
    )
    ref_words = reference.lower().split()
    hyp_words = hypothesis.lower().split()
    common_words = set(ref_words) & set(hyp_words)
    missing_words = set(ref_words) - set(hyp_words)
    extra_words = set(hyp_words) - set(ref_words)
    return {
        "wer": wer_score,
        "wer_percentage": wer_score * 100,
        "word_accuracy": 1 - wer_score,
        "total_ref_words": len(ref_words),
        "total_hyp_words": len(hyp_words),
        "common_words": common_words,
        "missing_words": missing_words,
        "extra_words": extra_words,
    }


def compute_rouge_metrics(reference: str, hypothesis: str, rouge: Rouge):
    try:
        scores = rouge.get_scores(hypothesis, reference)[0]
        return {
            "rouge-1": {
                "f": scores["rouge-1"]["f"] * 100,
                "p": scores["rouge-1"]["p"] * 100,
                "r": scores["rouge-1"]["r"] * 100,
            },
            "rouge-2": {
                "f": scores["rouge-2"]["f"] * 100,
                "p": scores["rouge-2"]["p"] * 100,
                "r": scores["rouge-2"]["r"] * 100,
            },
            "rouge-l": {
                "f": scores["rouge-l"]["f"] * 100,
                "p": scores["rouge-l"]["p"] * 100,
                "r": scores["rouge-l"]["r"] * 100,
            },
        }
    except Exception as e:
        st.error(f"Ошибка при расчете ROUGE: {e}")
        return None


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

# ============ AUDIO CONVERSION ============

def convert_to_wav(input_path, output_path=None):
    """Конвертирует аудио в WAV формат с помощью FFmpeg"""
    import subprocess
    
    if output_path is None:
        output_path = input_path.rsplit('.', 1)[0] + '_converted.wav'
    
    cmd = ['ffmpeg', '-i', input_path, '-ar', '16000', '-ac', '1', '-y', output_path]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    return output_path


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

        self.rouge = get_session_rouge()

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

    def calculate_wer(self, reference, hypothesis):
        """Вычисляет метрики WER и связанные метрики"""
        return compute_wer_metrics(reference, hypothesis)

    def calculate_rouge(self, reference, hypothesis):
        """Расчет ROUGE метрик"""
        return compute_rouge_metrics(reference, hypothesis, self.rouge)

    def evaluate_transcription(self, audio_path, reference_text=None):
        """Полная оценка транскрибации"""
        hypothesis = self.transcribe_audio(audio_path)
        result = {'hypothesis': hypothesis}
        
        if reference_text:
            wer_results = self.calculate_wer(reference_text, hypothesis)
            rouge_results = self.calculate_rouge(reference_text, hypothesis)
            result['wer'] = wer_results
            result['rouge'] = rouge_results
        
        return result


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


def create_protocol_docx(transcription):
    """Legacy: один блок «текст консультации» (режим разработчика)."""
    doc = Document()
    doc.add_heading("Протокол консультации", 0)
    doc.add_paragraph(f"Дата: {format_consultation_date_gmt3()}")
    doc.add_heading("Текст консультации", level=1)
    doc.add_paragraph(transcription)
    return doc


# ============ MAIN UI ============

st.title("🏥 ClinVoice")
st.markdown("**Распознавание речи для медицинских консультаций**")

# Check URL for dev mode
query_params = st.query_params
is_dev_mode = query_params.get("mode") == "dev"

# Show sidebar only in dev mode
if is_dev_mode:
    # Sidebar - Mode selection
    st.sidebar.title("Настройки")
    
    mode = st.sidebar.radio("Режим:", ["Doctor Mode", "Developer Mode"])
    
    st.sidebar.markdown("---")
    
    # Model selection - only show in dev modes (Doctor in dev still uses Hub fine-tuned only)
    if mode == "Doctor Mode":
        _hid = resolve_hub_model_id()
        model_source = f"HF fine-tuned ({_hid})"
        hub_model_id = _hid
        model_size = "small"
    else:
        model_source = st.sidebar.radio(
            "Источник модели:",
            ["HuggingFace", "Дообученная (мед., HF)"],
        )
        if model_source == "Дообученная (мед., HF)":
            hub_model_id = resolve_hub_model_id()
            model_size = "small"
        else:
            hub_model_id = None
            model_size = st.sidebar.selectbox(
                "Размер модели:",
                ["tiny", "base", "small", "medium", "large"],
                index=1,
            )
else:
    # Doctor Mode without sidebar — always medical fine-tuned from Hub
    mode = "Doctor Mode"
    _hid = resolve_hub_model_id()
    model_source = f"HF fine-tuned ({_hid})"
    hub_model_id = _hid
    model_size = "small"


# ============ DOCTOR MODE ============
if mode == "Doctor Mode":
    st.header("Запись консультации")
    st.warning(
        "При первом запуске загрузка и настройка приложения могут занять несколько минут — это нормально."
    )
    st.caption("Нажмите на микрофон, чтобы начать запись. Нажмите ещё раз, чтобы остановить.")
    
    # Initialize session state
    if "original_transcription" not in st.session_state:
        st.session_state.original_transcription = None
    if "protocol_editor_text" not in st.session_state:
        st.session_state.protocol_editor_text = ""

    audio_file = st.audio_input("🎙️ Запись", key="doctor_recorder")

    temp_path = "/tmp/recorded_audio.wav"
    if audio_file:
        with open(temp_path, "wb") as f:
            f.write(audio_file.getbuffer())
        st.success("Запись завершена! ✓")

        if st.button("Транскрибировать и заполнить протокол", type="primary"):
            if not yandex_llm_configured():
                st.error(
                    "Не заданы параметры для заполнения протокола: укажите "
                    "**YANDEX_FOLDER_ID** и (**YANDEX_CLOUD_API_KEY** или **YANDEX_IAM_TOKEN**) "
                    "в переменных окружения или в секретах приложения."
                )
                st.stop()

            with st.spinner("Транскрибация... Это может занять несколько минут."):
                transcriber = AudioTranscriberWithMetrics(
                    model_size=model_size, hub_model_id=hub_model_id, silent_ui=True
                )
                transcription = transcriber.transcribe_audio(temp_path)

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
                st.success("Готово. Проверьте и при необходимости отредактируйте протокол ниже.")
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
        col1, col2 = st.columns(2)
        doc = create_structured_protocol_docx(fields, consultation_date)
        doc_path = "/tmp/protocol.docx"
        doc.save(doc_path)
        with open(doc_path, "rb") as f:
            col1.download_button(
                "📄 Скачать .docx",
                f.read(),
                "protocol.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
        txt_body = build_structured_protocol_txt(fields, consultation_date)
        col2.download_button(
            "📄 Скачать .txt",
            txt_body,
            "protocol.txt",
            "text/plain",
        )


# ============ DEVELOPER MODE ============
elif mode == "Developer Mode":
    st.header("Загрузка данных")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Аудио")
        audio_file = st.file_uploader("Выберите аудиофайл (только WAV)", type=['wav'])
    
    with col2:
        st.subheader("Эталонный текст")
        ref_option = st.radio("Выберите способ:", ["Ввести вручную", "Загрузить .txt файл"], horizontal=True)
        
        reference_text = ""
        
        if ref_option == "Ввести вручную":
            reference_text = st.text_area("Эталонный текст:", height=100, key="ref_manual")
        else:
            ref_file = st.file_uploader("Загрузите файл с эталонным текстом:", type=['txt'], key="ref_file")
            if ref_file:
                reference_text = str(ref_file.read(), 'utf-8')
                st.success("Файл загружен!")
    
    if audio_file:
        temp_path = f"/tmp/{audio_file.name}"
        with open(temp_path, "wb") as f:
            f.write(audio_file.getbuffer())
        
        # Check if conversion is needed
        if audio_file.name.lower().endswith('.wav'):
            audio_path = temp_path
            st.info("Формат: WAV ✓")
        else:
            st.warning("Конвертация в WAV...")
            wav_path = temp_path.rsplit('.', 1)[0] + '.wav'
            audio_path = convert_to_wav(temp_path, wav_path)
            st.success("Конвертация завершена! ✓")
        
        st.audio(audio_file, format=audio_file.type)

        if "dev_protocol_editor_text" not in st.session_state:
            st.session_state.dev_protocol_editor_text = ""

        # Initialize session state for results
        if "transcription_done" not in st.session_state:
            st.session_state.transcription_done = False

        if st.button("Транскрибировать и заполнить протокол", type="primary"):
            if not yandex_llm_configured():
                st.error(
                    "Не заданы **YANDEX_FOLDER_ID** и ключ доступа (**YANDEX_CLOUD_API_KEY**) "
                    "или **YANDEX_IAM_TOKEN** в переменных окружения или секретах приложения."
                )
                st.stop()

            with st.spinner("Транскрибация..."):
                transcriber = AudioTranscriberWithMetrics(
                    model_size=model_size, hub_model_id=hub_model_id, silent_ui=True
                )
                transcription = transcriber.transcribe_audio(audio_path)

            st.session_state.transcription = transcription
            st.session_state.dev_transcript_editor = transcription
            st.session_state.transcription_done = True
            st.session_state.dev_protocol_consultation_date = format_consultation_date_gmt3()

            if reference_text:
                wer_results = transcriber.calculate_wer(reference_text, transcription)
                rouge_results = transcriber.calculate_rouge(reference_text, transcription)
                st.session_state.wer_results = wer_results
                st.session_state.rouge_results = rouge_results
            else:
                st.session_state.pop("wer_results", None)
                st.session_state.pop("rouge_results", None)

            try:
                with st.spinner("Заполнение протокола..."):
                    protocol = fill_protocol_from_transcript(transcription)
                st.session_state.dev_protocol_editor_text = format_protocol_editor_text(
                    st.session_state.dev_protocol_consultation_date,
                    protocol,
                )
                st.success("Готово. Проверьте транскрипт, метрики и протокол.")
            except Exception as e:
                st.error(f"Ошибка заполнения протокола: {e}")
                st.session_state.dev_protocol_editor_text = format_protocol_editor_text(
                    st.session_state.dev_protocol_consultation_date,
                    {k: "" for k in PROTOCOL_FIELD_KEYS},
                )

        # Show results if transcription was done
        if st.session_state.transcription_done:
            transcription = st.session_state.transcription

            # Show results
            st.header("Результаты")
            if "dev_transcript_editor" not in st.session_state:
                st.session_state.dev_transcript_editor = transcription
            with st.expander("Транскрипт (справочно, можно исправить)"):
                st.text_area(
                    "Транскрипт",
                    height=150,
                    key="dev_transcript_editor",
                    label_visibility="collapsed",
                )

            # Calculate and show metrics if reference text exists
            wer_results = None
            rouge_results = None
            if reference_text:
                if 'wer_results' not in st.session_state:
                    with st.spinner("Расчёт метрик..."):
                        st.session_state.wer_results = compute_wer_metrics(
                            reference_text, transcription
                        )
                        st.session_state.rouge_results = compute_rouge_metrics(
                            reference_text, transcription, get_session_rouge()
                        )
                
                wer_results = st.session_state.wer_results
                rouge_results = st.session_state.rouge_results
                
                st.subheader("Метрики качества")
                
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("WER", f"{wer_results['wer_percentage']:.2f}%")
                col2.metric("Word Accuracy", f"{(wer_results['word_accuracy']*100):.2f}%")
                col3.metric("Слов в эталоне", wer_results['total_ref_words'])
                col4.metric("Слов в гипотезе", wer_results['total_hyp_words'])
                
                if rouge_results:
                    st.subheader("ROUGE")
                    for rouge_type in ['rouge-1', 'rouge-2', 'rouge-l']:
                        scores = rouge_results[rouge_type]
                        st.write(f"**{rouge_type.upper()}**: F1={scores['f']:.2f}% | P={scores['p']:.2f}% | R={scores['r']:.2f}%")
                
                with st.expander("Детальный анализ слов"):
                    col1, col2 = st.columns(2)
                    col1.markdown("**Совпадающие:** " + ", ".join(wer_results['common_words']) if wer_results['common_words'] else "нет")
                    col2.markdown("**Пропущенные:** " + ", ".join(wer_results['missing_words']) if wer_results['missing_words'] else "нет")
                    st.markdown("**Лишние:** " + ", ".join(wer_results['extra_words']) if wer_results['extra_words'] else "нет")
            else:
                st.info("Добавьте эталонный текст выше, чтобы увидеть метрики")

            st.subheader("Протокол консультации")
            st.text_area(
                "Протокол (редактирование)",
                height=320,
                key="dev_protocol_editor_text",
                help="Формат: «Дата:», «Жалобы:», «Анамнез:», «Заключение:», «Рекомендации:».",
            )

            # Downloads
            st.subheader("Скачать отчёт (транскрипция и метрики)")

            col1, col2 = st.columns(2)
            
            # TXT report
            txt_content = f"""РЕЗУЛЬТАТЫ ТРАНСКРИБАЦИИ
========================

Текст консультации:
{transcription}
"""
            if reference_text and wer_results:
                txt_content += f"""
МЕТРИКИ КАЧЕСТВА
=================
WER: {wer_results['wer_percentage']:.2f}%
Word Accuracy: {(wer_results['word_accuracy']*100):.2f}%
Слов в эталоне: {wer_results['total_ref_words']}
Слов в гипотезе: {wer_results['total_hyp_words']}

Совпадающие: {', '.join(wer_results['common_words']) if wer_results['common_words'] else 'нет'}
Пропущенные: {', '.join(wer_results['missing_words']) if wer_results['missing_words'] else 'нет'}
Лишние: {', '.join(wer_results['extra_words']) if wer_results['extra_words'] else 'нет'}
"""
                if rouge_results:
                    txt_content += f"""
ROUGE-1: F1={rouge_results['rouge-1']['f']:.2f}% | P={rouge_results['rouge-1']['p']:.2f}% | R={rouge_results['rouge-1']['r']:.2f}%
ROUGE-2: F1={rouge_results['rouge-2']['f']:.2f}% | P={rouge_results['rouge-2']['p']:.2f}% | R={rouge_results['rouge-2']['r']:.2f}%
ROUGE-L: F1={rouge_results['rouge-l']['f']:.2f}% | P={rouge_results['rouge-l']['p']:.2f}% | R={rouge_results['rouge-l']['r']:.2f}%
"""
            
            txt_path = "/tmp/report.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(txt_content)
            with open(txt_path, "r", encoding="utf-8") as f:
                col1.download_button(
                    "Скачать отчёт (.txt)",
                    f.read(),
                    "report.txt",
                    "text/plain",
                    key="dev_asr_report_txt",
                )

            # DOCX (legacy: один блок текста)
            doc = create_protocol_docx(transcription)
            doc_path = "/tmp/report.docx"
            doc.save(doc_path)
            with open(doc_path, "rb") as f:
                col2.download_button(
                    "Скачать отчёт (.docx)",
                    f.read(),
                    "report.docx",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key="dev_asr_report_docx",
                )

            st.subheader("Скачать протокол (как у врача)")
            _dev_parsed_date, dev_fields = parse_protocol_editor_text(
                st.session_state.get("dev_protocol_editor_text", "")
            )
            dev_consultation_date = (
                _dev_parsed_date.strip()
                or st.session_state.get("dev_protocol_consultation_date")
                or format_consultation_date_gmt3()
            )
            sdoc = create_structured_protocol_docx(dev_fields, dev_consultation_date)
            sdoc_path = "/tmp/dev_protocol.docx"
            sdoc.save(sdoc_path)
            col3, col4 = st.columns(2)
            with open(sdoc_path, "rb") as f:
                col3.download_button(
                    "📄 Протокол .docx",
                    f.read(),
                    "protocol_dev.docx",
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    key="dev_structured_docx",
                )
            dev_txt_body = build_structured_protocol_txt(dev_fields, dev_consultation_date)
            col4.download_button(
                "📄 Протокол .txt",
                dev_txt_body,
                "protocol_dev.txt",
                "text/plain",
                key="dev_structured_txt",
            )

        if os.path.exists(audio_path):
            os.remove(audio_path)

# Footer
st.markdown("---")
st.markdown("*ClinVoice v0.3 - Medical ASR Tool*")
