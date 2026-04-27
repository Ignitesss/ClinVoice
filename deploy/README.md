# Развёртывание ClinVoice на ВМ

Примеры конфигурации:

- `nginx-clinvoice.conf.example` — reverse proxy и TLS (Let’s Encrypt).
- `clinvoice.service.example` — systemd unit для `streamlit run` на `127.0.0.1:8501`.

Переменные окружения приложения см. в коде (`CLINVOICE_*`): кэш, Whisper, SQLite (`CLINVOICE_SQLITE_PATH`, `CLINVOICE_DB_TTL_HOURS`), WebRTC ICE JSON.
