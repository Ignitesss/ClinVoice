# ClinVoice - Распознавание речи для медицинских консультаций

A Streamlit app for doctors to transcribe patient consultations with Active Speech Recognition.

## Setup

```bash
pip install -r requirements.txt

# Make sure FFmpeg is installed
# Ubuntu: sudo apt install ffmpeg
# macOS: brew install ffmpeg

# Run the app
streamlit run app.py
```

## Возможности

### 1. Режим врача
- Простой текст + ключевые слова
- Скачиваемый протокол в .docx
- Для врачей, которым нужна просто транскрибация

### 2. Режим разработчика
- WER (Word Error Rate)
- ROUGE-1, ROUGE-2, ROUGE-L метрики
- Приоритетные ключевые слова
- Детальный анализ слов (совпадающие, пропущенные, лишние)
- Для тестирования и оценки качества модели

### 3. Сравнение моделей
- Сравнение двух моделей Whisper
- Разница в WER
- Сравнение ключевых слов

## Requirements

- Python 3.8+
- FFmpeg (system)
- ~2GB disk space for Whisper models

## Usage

1. Select mode from sidebar
2. Upload audio file
3. Click "Transcribe" / "Compare" / "Fine-tune"
4. Download results

## Project Structure

```
workspace/
├── app.py           # Main Streamlit app
├── requirements.txt # Dependencies
├── README.md        # This file
```

## For Team

- **Yana** works on UI and integration
- **Teammate** fine-tunes Whisper with medical terms → provides model
- Both use Developer Mode for testing
# ClinVoice
