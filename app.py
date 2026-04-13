# -*- coding: utf-8 -*-
"""
ClinVoice - Medical Speech Recognition App
Full version with AudioTranscriber, WhisperFineTuner, and ModelComparator
"""

import streamlit as st
import whisper
import jiwer
from rouge import Rouge
from docx import Document
import os
import torch
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from metrics_sheets import (
    build_metrics_row,
    metrics_sheets_secrets_configured,
    submit_metrics_row_to_sheets,
)

from protocol import (
    PROTOCOL_FIELD_KEYS,
    fill_protocol_from_transcript,
    resolve_openai_api_key,
    resolve_openai_model,
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


def resolve_hf_token() -> Optional[str]:
    t = (os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or "").strip()
    return t or None


def infer_whisper_processor_repo(hub_model_id: str, token: Optional[str]) -> str:
    """Базовый openai/whisper-* для процессора, если в finetune-репо нет preprocessor_config.json / tokenizer."""
    explicit = (os.environ.get("CLINVOICE_WHISPER_BASE_REPO") or "").strip()
    if explicit:
        return explicit
    try:
        cfg = AutoConfig.from_pretrained(hub_model_id, token=token)
        cand = getattr(cfg, "_name_or_path", None) or getattr(cfg, "name_or_path", None)
        if cand:
            s = str(cand).strip()
            if s.startswith("openai/whisper-"):
                return s
    except Exception:
        pass
    return "openai/whisper-small"


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
        """Whisper: либо базовая openai-whisper (hub_model_id=None), либо дообученная с Hugging Face Hub."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        token = resolve_hf_token()
        safe_hub = (hub_model_id or "").replace("/", "_")
        cache_key = f"asr_{model_size}_{safe_hub or 'openai_whisper'}"
        if cache_key not in st.session_state:
            if hub_model_id:
                if not TRANSFORMERS_AVAILABLE:
                    st.error("Пакет transformers необходим для дообученной модели с Hub.")
                    st.stop()
                if not silent_ui:
                    st.info(
                        f"Загрузка дообученной модели с Hugging Face: {hub_model_id} "
                        f"(первый запуск может занять несколько минут)..."
                    )
                try:
                    kwargs = {"torch_dtype": torch.float32}
                    if token:
                        kwargs["token"] = token
                    self.model = WhisperForConditionalGeneration.from_pretrained(hub_model_id, **kwargs)
                    proc_kwargs = {"token": token} if token else {}
                    try:
                        self.processor = WhisperProcessor.from_pretrained(hub_model_id, **proc_kwargs)
                    except Exception:
                        base_proc = infer_whisper_processor_repo(hub_model_id, token)
                        if not silent_ui:
                            st.warning(
                                f"В репозитории «{hub_model_id}» нет полного процессора "
                                f"(нужен preprocessor_config.json и файлы токенизатора). "
                                f"Загружаю процессор с **{base_proc}**. "
                                f"Если размер не совпадает с дообучением, задайте переменную **CLINVOICE_WHISPER_BASE_REPO** "
                                f"(например, `openai/whisper-base`)."
                            )
                        self.processor = WhisperProcessor.from_pretrained(base_proc, **proc_kwargs)
                    self.feature_extractor = self.processor.feature_extractor
                    self.tokenizer = self.processor.tokenizer
                    self.use_transformers = True
                    self.model.to(self.device)
                    self.model.eval()
                    st.session_state[cache_key] = {
                        'model': self.model,
                        'processor': self.processor,
                        'feature_extractor': self.feature_extractor,
                        'tokenizer': self.tokenizer,
                        'use_transformers': True,
                    }
                except Exception as e:
                    st.error(f"Ошибка загрузки модели с Hub: {e}")
                    st.stop()
            else:
                if not silent_ui:
                    st.info(f"Загрузка базовой модели Whisper ({model_size})...")
                self.model = whisper.load_model(model_size)
                self.use_transformers = False
                st.session_state[cache_key] = {'model': self.model, 'use_transformers': False}
        else:
            # Load from cache
            cached = st.session_state[cache_key]
            self.model = cached['model']
            self.use_transformers = cached.get('use_transformers', False)
            if self.use_transformers:
                self.processor = cached['processor']
                self.feature_extractor = cached['feature_extractor']
                self.tokenizer = cached['tokenizer']
                self.model.to(self.device)
                self.model.eval()
        
        self.rouge = Rouge()

    def transcribe_audio(self, audio_path, language='ru'):
        """Транскрибация аудиофайла"""
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
        transformation = jiwer.Compose([
            jiwer.ToLowerCase(),
            jiwer.RemoveWhiteSpace(replace_by_space=True),
            jiwer.RemoveMultipleSpaces(),
            jiwer.Strip(),
            jiwer.RemovePunctuation(),
            jiwer.ReduceToListOfListOfWords()
        ])
        
        wer_score = jiwer.wer(
            reference, hypothesis,
            reference_transform=transformation,
            hypothesis_transform=transformation
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
            "extra_words": extra_words
        }

    def calculate_rouge(self, reference, hypothesis):
        """Расчет ROUGE метрик"""
        try:
            scores = self.rouge.get_scores(hypothesis, reference)[0]
            return {
                'rouge-1': {'f': scores['rouge-1']['f'] * 100, 'p': scores['rouge-1']['p'] * 100, 'r': scores['rouge-1']['r'] * 100},
                'rouge-2': {'f': scores['rouge-2']['f'] * 100, 'p': scores['rouge-2']['p'] * 100, 'r': scores['rouge-2']['r'] * 100},
                'rouge-l': {'f': scores['rouge-l']['f'] * 100, 'p': scores['rouge-l']['p'] * 100, 'r': scores['rouge-l']['r'] * 100}
            }
        except Exception as e:
            st.error(f"Ошибка при расчете ROUGE: {e}")
            return None

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


class ModelComparator:
    """Класс для сравнения двух моделей"""
    
    def __init__(self, model1: AudioTranscriberWithMetrics, model2: AudioTranscriberWithMetrics,
                 model1_name: str = "Базовая модель", model2_name: str = "Дообученная модель"):
        self.model1 = model1
        self.model2 = model2
        self.model1_name = model1_name
        self.model2_name = model2_name

    def compare_on_file(self, audio_path: str, reference_text: str):
        """Сравнение двух моделей на одном аудиофайле"""
        st.subheader(f"--- {self.model1_name} ---")
        result1 = self.model1.evaluate_transcription(audio_path, reference_text)
        
        st.subheader(f"--- {self.model2_name} ---")
        result2 = self.model2.evaluate_transcription(audio_path, reference_text)
        
        # Сравнение WER
        if 'wer' in result1 and 'wer' in result2:
            col1, col2 = st.columns(2)
            with col1:
                st.metric(f"WER {self.model1_name}", f"{result1['wer']['wer_percentage']:.2f}%")
            with col2:
                st.metric(f"WER {self.model2_name}", f"{result2['wer']['wer_percentage']:.2f}%")
            
            wer_diff = result2['wer']['wer_percentage'] - result1['wer']['wer_percentage']
            if wer_diff < 0:
                st.success(f"{self.model2_name} лучше на {abs(wer_diff):.2f}%")
            elif wer_diff > 0:
                st.success(f"{self.model1_name} лучше на {abs(wer_diff):.2f}%")
        
        return {'model1': result1, 'model2': result2}


# ============ DOCX / TXT GENERATION ============

def format_consultation_date_gmt3() -> str:
    """Дата/время в поясе Europe/Moscow (UTC+3)."""
    return datetime.now(ZoneInfo("Europe/Moscow")).strftime("%d.%m.%Y %H:%M (GMT+3)")


def create_structured_protocol_docx(fields: dict, consultation_date: str, metadata=None):
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
    if metadata:
        doc.add_heading("Метаданные", level=1)
        for key, value in metadata.items():
            doc.add_paragraph(f"{key}: {value}")
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


def create_protocol_docx(transcription, metadata=None):
    """Legacy: один блок «текст консультации» (Developer / сравнение моделей)."""
    doc = Document()
    doc.add_heading("Протокол консультации", 0)
    doc.add_paragraph(f"Дата: {format_consultation_date_gmt3()}")
    doc.add_heading("Текст консультации", level=1)
    doc.add_paragraph(transcription)
    if metadata:
        doc.add_heading("Метаданные", level=1)
        for key, value in metadata.items():
            doc.add_paragraph(f"{key}: {value}")
    return doc


# ============ MAIN UI ============

st.title("🏥 ClinVoice")
st.markdown("**Распознавание речи для медицинских консультаций**")

# Check URL for dev mode
query_params = st.query_params
is_dev_mode = query_params.get("mode") == "dev"
metrics_debug = query_params.get("metrics_debug") == "1"

# Show sidebar only in dev mode
if is_dev_mode:
    # Sidebar - Mode selection
    st.sidebar.title("Настройки")
    
    mode = st.sidebar.radio("Режим:", 
        ["Doctor Mode", "Developer Mode", "Model Comparison"])
    
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
    if metrics_debug and not metrics_sheets_secrets_configured():
        st.info(
            "Отладка метрик (`metrics_debug=1`): не заданы `CLINVOICE_SHEETS_WEBAPP_URL` "
            "или `CLINVOICE_SHEETS_SECRET` в Secrets / окружении."
        )
    st.warning(
        "При первом запуске загрузка и настройка приложения могут занять несколько минут — это нормально."
    )
    st.caption("Нажмите на микрофон, чтобы начать запись. Нажмите ещё раз, чтобы остановить.")
    
    # Initialize session state
    if "original_transcription" not in st.session_state:
        st.session_state.original_transcription = None
    if "doctor_metrics_saved" not in st.session_state:
        st.session_state.doctor_metrics_saved = False
    for _fk in PROTOCOL_FIELD_KEYS:
        _sk = f"proto_{_fk}"
        if _sk not in st.session_state:
            st.session_state[_sk] = ""

    audio_file = st.audio_input("🎙️ Запись", key="doctor_recorder")

    temp_path = "/tmp/recorded_audio.wav"
    if audio_file:
        with open(temp_path, "wb") as f:
            f.write(audio_file.getbuffer())
        st.success("Запись завершена! ✓")

        if st.button("Транскрибировать и заполнить протокол", type="primary"):
            api_key = resolve_openai_api_key()
            if not api_key:
                st.error(
                    "Не задан **OPENAI_API_KEY**: добавьте ключ в переменные окружения "
                    "или в Streamlit Secrets."
                )
                st.stop()

            st.session_state.doctor_metrics_saved = False

            with st.spinner("Транскрибация... Это может занять несколько минут."):
                transcriber = AudioTranscriberWithMetrics(
                    model_size=model_size, hub_model_id=hub_model_id, silent_ui=True
                )
                transcription = transcriber.transcribe_audio(temp_path)

            st.session_state.original_transcription = transcription
            st.session_state.doctor_transcript_editor = transcription
            st.session_state.protocol_consultation_date = format_consultation_date_gmt3()

            try:
                with st.spinner("Заполнение протокола (ИИ)..."):
                    protocol = fill_protocol_from_transcript(
                        transcription,
                        api_key,
                        model=resolve_openai_model(),
                    )
                for _k in PROTOCOL_FIELD_KEYS:
                    st.session_state[f"proto_{_k}"] = protocol[_k]
                st.success("Готово. Проверьте и при необходимости отредактируйте поля протокола.")
            except Exception as e:
                st.error(f"Ошибка заполнения протокола (OpenAI): {e}")
                for _k in PROTOCOL_FIELD_KEYS:
                    _sk = f"proto_{_k}"
                    if _sk not in st.session_state:
                        st.session_state[_sk] = ""

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

        edited_tr = st.session_state.get("doctor_transcript_editor", st.session_state.original_transcription)
        if (
            edited_tr != st.session_state.original_transcription
            and not st.session_state.doctor_metrics_saved
        ):
            transcriber = AudioTranscriberWithMetrics(
                model_size=model_size, hub_model_id=hub_model_id, silent_ui=True
            )
            wer_results = transcriber.calculate_wer(edited_tr, st.session_state.original_transcription)
            rouge_results = transcriber.calculate_rouge(edited_tr, st.session_state.original_transcription)
            row = build_metrics_row(
                wer_results,
                rouge_results,
                resolve_hub_model_id(),
            )
            _ok, _metrics_err = submit_metrics_row_to_sheets(row)
            if metrics_debug:
                if _ok:
                    st.success("Метрики: отправка успешна (режим отладки `metrics_debug=1`).")
                else:
                    st.warning(
                        "Метрики: отправка не удалась — "
                        + (_metrics_err or "неизвестная ошибка")
                    )
            st.session_state.doctor_metrics_saved = True

        _labels = {
            "complaints": "Жалобы",
            "anamnesis": "Анамнез",
            "conclusion": "Заключение",
            "recommendations": "Рекомендации",
        }
        for _k in PROTOCOL_FIELD_KEYS:
            st.text_area(_labels[_k], key=f"proto_{_k}", height=120)

        consultation_date = st.session_state.get("protocol_consultation_date") or format_consultation_date_gmt3()
        fields = {k: st.session_state.get(f"proto_{k}", "") for k in PROTOCOL_FIELD_KEYS}

        st.subheader("Скачать протокол")
        col1, col2 = st.columns(2)
        meta = {"model_whisper": resolve_hub_model_id(), "model_llm": resolve_openai_model()}
        doc = create_structured_protocol_docx(fields, consultation_date, metadata=meta)
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

        for _fk in PROTOCOL_FIELD_KEYS:
            _sk = f"dev_proto_{_fk}"
            if _sk not in st.session_state:
                st.session_state[_sk] = ""

        # Initialize session state for results
        if "transcription_done" not in st.session_state:
            st.session_state.transcription_done = False

        if st.button("Транскрибировать и заполнить протокол", type="primary"):
            api_key = resolve_openai_api_key()
            if not api_key:
                st.error(
                    "Не задан **OPENAI_API_KEY**: добавьте ключ в переменные окружения "
                    "или в Streamlit Secrets."
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
                with st.spinner("Заполнение протокола (ИИ)..."):
                    protocol = fill_protocol_from_transcript(
                        transcription,
                        api_key,
                        model=resolve_openai_model(),
                    )
                for _k in PROTOCOL_FIELD_KEYS:
                    st.session_state[f"dev_proto_{_k}"] = protocol[_k]
                st.success("Готово. Проверьте транскрипт, метрики и поля протокола.")
            except Exception as e:
                st.error(f"Ошибка заполнения протокола (OpenAI): {e}")
                for _k in PROTOCOL_FIELD_KEYS:
                    _sk = f"dev_proto_{_k}"
                    if _sk not in st.session_state:
                        st.session_state[_sk] = ""

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
                        transcriber = AudioTranscriberWithMetrics(model_size=model_size, hub_model_id=hub_model_id)
                        st.session_state.wer_results = transcriber.calculate_wer(reference_text, transcription)
                        st.session_state.rouge_results = transcriber.calculate_rouge(reference_text, transcription)
                
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

            _dev_labels = {
                "complaints": "Жалобы",
                "anamnesis": "Анамнез",
                "conclusion": "Заключение",
                "recommendations": "Рекомендации",
            }
            st.subheader("Протокол консультации (ИИ)")
            for _k in PROTOCOL_FIELD_KEYS:
                st.text_area(_dev_labels[_k], key=f"dev_proto_{_k}", height=120)

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
            doc = create_protocol_docx(transcription, {"model": model_size})
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
            dev_fields = {k: st.session_state.get(f"dev_proto_{k}", "") for k in PROTOCOL_FIELD_KEYS}
            dev_consultation_date = (
                st.session_state.get("dev_protocol_consultation_date") or format_consultation_date_gmt3()
            )
            dev_meta = {
                "model_whisper": hub_model_id or f"openai-whisper-{model_size}",
                "model_llm": resolve_openai_model(),
                "mode": "Developer Mode",
            }
            sdoc = create_structured_protocol_docx(dev_fields, dev_consultation_date, metadata=dev_meta)
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


# ============ MODEL COMPARISON MODE ============
elif mode == "Model Comparison":
    st.header("Сравнение моделей")
    
    # Model 1 selection
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Модель 1")
        model1_source = st.radio("Источник:", ["HuggingFace", "Дообученная (мед., HF)"], key="m1_src")
        if model1_source == "Дообученная (мед., HF)":
            model1_hub_id = resolve_hub_model_id()
            model1_size = "small"
        else:
            model1_hub_id = None
            model1_size = st.selectbox("Размер:", ["tiny", "base", "small", "medium", "large"], index=1, key="m1_sz")
    
    with col2:
        st.subheader("Модель 2")
        model2_source = st.radio("Источник:", ["HuggingFace", "Дообученная (мед., HF)"], key="m2_src")
        if model2_source == "Дообученная (мед., HF)":
            model2_hub_id = resolve_hub_model_id()
            model2_size = "small"
        else:
            model2_hub_id = None
            model2_size = st.selectbox("Размер:", ["tiny", "base", "small", "medium", "large"], index=4, key="m2_sz")
    
    st.subheader("Загрузка аудио")
    audio_file = st.file_uploader("Выберите аудиофайл (только WAV)", type=['wav'])
    
    # Initialize session state for comparison
    if 'comparison_done' not in st.session_state:
        st.session_state.comparison_done = False
    
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
        
        # Reference text
        ref_option = st.radio("Эталонный текст:", ["Ввести вручную", "Загрузить .txt файл"], horizontal=True, key="ref_comp")
        
        reference_text = ""
        
        if ref_option == "Ввести вручную":
            reference_text = st.text_area("Эталонный текст:", height=100)
        else:
            ref_file = st.file_uploader("Загрузите файл с эталонным текстом:", type=['txt'], key="ref_file_comp")
            if ref_file:
                reference_text = str(ref_file.read(), 'utf-8')
                st.success("Файл загружен!")
        
        if st.button("Сравнить модели", type="primary"):
            with st.spinner("Запуск сравнения..."):
                model1 = AudioTranscriberWithMetrics(model_size=model1_size, hub_model_id=model1_hub_id)
                model2 = AudioTranscriberWithMetrics(model_size=model2_size, hub_model_id=model2_hub_id)
                
                model1_name = f"Модель 1 ({model1_source})"
                model2_name = f"Модель 2 ({model2_source})"
                
                comparator = ModelComparator(
                    model1, model2,
                    model1_name,
                    model2_name
                )
                
                results = comparator.compare_on_file(audio_path, reference_text)
                
                # Save to session state
                st.session_state.comparison_done = True
                st.session_state.comparison_results = results
                st.session_state.model1_name = model1_name
                st.session_state.model2_name = model2_name
        
        # Show results and download buttons if comparison was done
        if st.session_state.comparison_done and 'comparison_results' in st.session_state:
            results = st.session_state.comparison_results
            model1_name = st.session_state.model1_name
            model2_name = st.session_state.model2_name
            
            # Add download buttons
            if results and reference_text:
                st.subheader("Скачать отчёт")
                
                result1 = results['model1']
                result2 = results['model2']
                
                txt_content = f"""ОТЧЁТ ПО СРАВНЕНИЮ МОДЕЛЕЙ
========================

{model1_name}:
- Текст: {result1['hypothesis']}
- WER: {result1['wer']['wer_percentage']:.2f}%

{model2_name}:
- Текст: {result2['hypothesis']}
- WER: {result2['wer']['wer_percentage']:.2f}%

Итог: {'Модель 1 лучше' if result1['wer']['wer_percentage'] < result2['wer']['wer_percentage'] else 'Модель 2 лучше'}
"""
                
                txt_path = "/tmp/comparison_report.txt"
                with open(txt_path, "w", encoding="utf-8") as f:
                    f.write(txt_content)
                with open(txt_path, "r", encoding="utf-8") as f:
                    st.download_button("Скачать отчёт (.txt)", f.read(), "comparison_report.txt", "text/plain", key="comp_txt")
        
        if os.path.exists(audio_path):
            os.remove(audio_path)

# Footer
st.markdown("---")
st.markdown("*ClinVoice v0.3 - Medical ASR Tool*")
