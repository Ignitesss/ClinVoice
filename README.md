# ClinVoice — распознавание речи и протокол консультации

Приложение для врачей: запись **одним потоком** с 
микрофона; аудио накапливается для нашей 
дооубченной модели.
Просто говорите - наше приложение само сделаем 
протокол.

Архитектура: **Vue 3** (SPA) + **FastAPI** + **nginx** (статика, прокси `/api` и `/ws`).

## Требования

- Python **3.10–3.12** (рекомендуется **3.12**)
- Node.js **18+** для сборки фронтенда
- ~2 GB места под модели Whisper / Hugging Face

## Установка (разработка)

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cd frontend && npm install && npm run build && cd ..
```

Переменные окружения (минимум для входа и БД):

- `CLINVOICE_JWT_SECRET` — секрет подписи cookie-сессии (в продакшене обязателен).
- `CLINVOICE_SQLITE_PATH` — путь к SQLite (по умолчанию см. `auth.resolve_sqlite_path_cli`).

Для SpeechKit и YandexGPT задайте `YANDEX_FOLDER_ID` и `YANDEX_CLOUD_API_KEY` или `YANDEX_IAM_TOKEN` (см. комментарии в `protocol.py`).

Создание пользователя (один раз):

```bash
.venv/bin/python scripts/create_user.py USER
```

Пароль запросится в терминале; для скриптов можно задать `CLINVOICE_BOOTSTRAP_PASSWORD`.

Запуск API:

```bash
./scripts/run_api.sh
# или: .venv/bin/python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --workers 1
```

Фронтенд в режиме разработки (прокси на API в `frontend/vite.config.ts`):

```bash
cd frontend && npm run dev
```

Откройте интерфейс по адресу Vite (обычно http://localhost:5173), войдите под созданным пользователем.

## Продакшен (nginx + systemd)

1. Соберите фронт: `cd frontend && npm run build`.
2. Скопируйте репозиторий (или артефакты) на сервер, активируйте venv и `pip install -r requirements.txt`.
3. Пример **nginx**: `deploy/nginx-clinvoice.conf.example` — статика из `frontend/dist`, `proxy_pass` на `127.0.0.1:8000` для `/api/` и `/ws/` с заголовками Upgrade для WebSocket.
4. Пример **systemd**: `deploy/clinvoice-api.service.example` — `uvicorn` с **`--workers 1`** из-за SQLite.
5. **Cloudflare Tunnel**: убедитесь, что туннель ведёт на nginx (или напрямую на uvicorn), и что на тарифе разрешены долгие WebSocket и достаточные таймауты; при необходимости увеличьте `proxy_read_timeout` в nginx.

## Как пользоваться

1. После входа создаётся консультация; ссылку с идентификатором можно сохранить.
2. Включите запись с микрофона; при необходимости включите паузу.
3. Нажмите **«Заполнить протокол»** — и получите готовый протокол.
4. Отредактируйте блок протокола и при необходимости сохраните черновик в БД.

## Команда

- **Яна** — разработка и интеграция, тестирование
- **Любовь** — дообучение Whisper на медицинском словаре, тестирование
