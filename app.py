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
import re
from collections import Counter
import torch
import numpy as np
from typing import List

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
    def __init__(self, model_size="base", model_path=None):
        """Инициализация модели Whisper"""
        # Use session state to cache model
        cache_key = f"whisper_{model_size}_{model_path or 'base'}"
        if cache_key not in st.session_state:
            st.session_state[cache_key] = whisper.load_model(model_size)
        self.model = st.session_state[cache_key]
        
        self.rouge = Rouge()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def transcribe_audio(self, audio_path, language='ru'):
        """Транскрибация аудиофайла"""
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

    def extract_keywords(self, text, top_n=10, custom_keywords=None):
        """Извлечение ключевых слов с поддержкой пользовательских"""
        words = re.findall(r'\b\w+\b', text.lower())
        
        stop_words = {
            'и', 'в', 'на', 'с', 'по', 'о', 'у', 'к', 'а', 'но', 'за', 'из',
            'от', 'до', 'не', 'что', 'это', 'как', 'так', 'для', 'то', 'же',
            'бы', 'был', 'была', 'было', 'были', 'его', 'ее', 'их', 'мне', 'тебе',
            'нас', 'им', 'меня', 'тебя', 'вас', 'нас', 'их', 'наш', 'ваш',
            'свой', 'мой', 'твой', 'этот', 'эта', 'это', 'эти', 'весь', 'вся',
            'все', 'всё', 'всех', 'всем', 'веми'
        }
        
        filtered_words = [w for w in words if w not in stop_words and len(w) > 2]
        word_counts = Counter(filtered_words)
        
        if custom_keywords:
            custom_counts = Counter()
            for word in custom_keywords:
                if word.lower() in filtered_words:
                    custom_counts[word.lower()] = word_counts[word.lower()] + 1000
            word_counts = custom_counts + word_counts
        
        return [word for word, count in word_counts.most_common(top_n)]

    def evaluate_transcription(self, audio_path, reference_text=None, custom_keywords=None):
        """Полная оценка транскрибации"""
        hypothesis = self.transcribe_audio(audio_path)
        keywords = self.extract_keywords(hypothesis, 15, custom_keywords)
        
        result = {'hypothesis': hypothesis, 'keywords': keywords}
        
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

    def compare_on_file(self, audio_path: str, reference_text: str, custom_keywords: List[str] = None):
        """Сравнение двух моделей на одном аудиофайле"""
        st.subheader(f"--- {self.model1_name} ---")
        result1 = self.model1.evaluate_transcription(audio_path, reference_text, custom_keywords)
        
        st.subheader(f"--- {self.model2_name} ---")
        result2 = self.model2.evaluate_transcription(audio_path, reference_text, custom_keywords)
        
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
        
        # Ключевые слова
        st.write(f"**Ключевые слова {self.model1_name}**: {', '.join(result1['keywords'][:5])}")
        st.write(f"**Ключевые слова {self.model2_name}**: {', '.join(result2['keywords'][:5])}")
        
        return {'model1': result1, 'model2': result2}


# ============ DOCX GENERATION ============

def create_protocol_docx(transcription, keywords=None, metadata=None):
    doc = Document()
    doc.add_heading('Протокол консультации', 0)
    doc.add_paragraph(f'Дата: {os.popen("date +%d.%m.%Y").read().strip()}')
    
    doc.add_heading('Текст консультации', level=1)
    doc.add_paragraph(transcription)
    
    if keywords:
        doc.add_heading('Ключевые слова', level=1)
        doc.add_paragraph(', '.join(keywords))
    
    if metadata:
        doc.add_heading('Метаданные', level=1)
        for key, value in metadata.items():
            doc.add_paragraph(f'{key}: {value}')
    
    return doc


# ============ MAIN UI ============

st.title("🏥 ClinVoice")
st.markdown("**Распознавание речи для медицинских консультаций**")

# Sidebar - Mode selection
st.sidebar.title("Настройки")
mode = st.sidebar.radio("Режим:", 
    ["Doctor Mode", "Developer Mode", "Model Comparison"])

st.sidebar.markdown("---")
model_size = st.sidebar.selectbox("Размер модели:", 
    ["tiny", "base", "small", "medium", "large"], index=1)


# ============ DOCTOR MODE ============
if mode == "Doctor Mode":
    st.header("Загрузка аудио")
    audio_file = st.file_uploader("Выберите аудиофайл (только WAV)", type=['wav'])
    
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
        
        if st.button("Транскрибировать", type="primary"):
            with st.spinner("Транскрибация..."):
                transcriber = AudioTranscriberWithMetrics(model_size=model_size)
                transcription = transcriber.transcribe_audio(audio_path)
                keywords = transcriber.extract_keywords(transcription, 15)
            
            st.success("Готово!")
            
            st.header("Протокол консультации")
            st.text_area("Текст:", transcription, height=200, key="doctor_text")
            st.subheader("Ключевые слова")
            st.write(", ".join(keywords))
            
            # Downloads
            col1, col2 = st.columns(2)
            doc = create_protocol_docx(transcription, keywords)
            doc_path = "/tmp/protocol.docx"
            doc.save(doc_path)
            with open(doc_path, "rb") as f:
                col1.download_button("Скачать .docx", f.read(), "protocol.docx",
                                   "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
            
            txt_path = "/tmp/protocol.txt"
            with open(txt_path, "w", encoding="utf-8") as f:
                f.write(transcription)
            with open(txt_path, "r", encoding="utf-8") as f:
                col2.download_button("Скачать .txt", f.read(), "protocol.txt", "text/plain")
        
        if os.path.exists(audio_path):
            os.remove(audio_path)


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
    
    # Custom keywords
    st.subheader("Ключевые слова")
    custom_kw_input = st.text_input("Приоритетные слова (через запятую):", 
                                  placeholder="гипертрофия, тахикардия", key="custom_kw")
    custom_keywords = [kw.strip().lower() for kw in custom_kw_input.split(',') if kw.strip()]
    
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
        
        if st.button("Транскрибировать", type="primary"):
            with st.spinner("Транскрибация..."):
                transcriber = AudioTranscriberWithMetrics(model_size=model_size)
                transcription = transcriber.transcribe_audio(audio_path)
            
            st.success("Готово!")
            
            # Extract keywords
            if custom_keywords:
                keywords = transcriber.extract_keywords(transcription, 15, custom_keywords)
            else:
                keywords = transcriber.extract_keywords(transcription, 15)
            
            # Show results
            st.header("Результаты")
            st.text_area("Текст:", transcription, height=150, key="dev_text")
            
            # Show keywords
            st.subheader("Ключевые слова")
            st.write(", ".join(keywords))
            
            # Calculate and show metrics if reference text exists
            if reference_text:
                with st.spinner("Расчёт метрик..."):
                    wer_results = transcriber.calculate_wer(reference_text, transcription)
                    rouge_results = transcriber.calculate_rouge(reference_text, transcription)
                
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
            
            # Downloads
            st.subheader("Скачать отчёт")
            
            col1, col2 = st.columns(2)
            
            # TXT report
            txt_content = f"""РЕЗУЛЬТАТЫ ТРАНСКРИБАЦИИ
========================

Текст консультации:
{transcription}

Ключевые слова: {', '.join(keywords)}
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
                col1.download_button("Скачать отчёт (.txt)", f.read(), "report.txt", "text/plain")
            
            # DOCX
            doc = create_protocol_docx(transcription, keywords, {"model": model_size})
            doc_path = "/tmp/report.docx"
            doc.save(doc_path)
            with open(doc_path, "rb") as f:
                col2.download_button("Скачать протокол (.docx)", f.read(), "report.docx",
                                   "application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        
        if os.path.exists(audio_path):
            os.remove(audio_path)


# ============ MODEL COMPARISON MODE ============
elif mode == "Model Comparison":
    st.header("Сравнение моделей")
    
    col1, col2 = st.columns(2)
    model1_size = col1.selectbox("Модель 1:", ["tiny", "base", "small", "medium", "large"], index=1, key="m1")
    model2_size = col2.selectbox("Модель 2:", ["tiny", "base", "small", "medium", "large"], index=4, key="m2")
    
    st.subheader("Загрузка аудио")
    audio_file = st.file_uploader("Выберите аудиофайл (только WAV)", type=['wav'])
    
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
        
        # Custom keywords
        custom_kw_input = st.text_input("Приоритетные слова (через запятую):", 
                                        placeholder="гипертрофия, тахикардия")
        custom_keywords = [kw.strip().lower() for kw in custom_kw_input.split(',') if kw.strip()]
        
        if st.button("Сравнить модели", type="primary"):
            with st.spinner("Запуск сравнения..."):
                model1 = AudioTranscriberWithMetrics(model_size=model1_size)
                model2 = AudioTranscriberWithMetrics(model_size=model2_size)
                
                comparator = ModelComparator(
                    model1, model2,
                    f"Whisper {model1_size}",
                    f"Whisper {model2_size}"
                )
                
                comparator.compare_on_file(audio_path, reference_text, custom_keywords)
        
        if os.path.exists(audio_path):
            os.remove(audio_path)

# Footer
st.markdown("---")
st.markdown("*ClinVoice v0.3 - Medical ASR Tool*")
