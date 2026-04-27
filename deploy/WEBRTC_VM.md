# WebRTC на собственной ВМ (HTTPS и ICE)

ClinVoice использует микрофон через **WebRTC** (`streamlit-webrtc`). На продакшене это завязано на следующее.

## HTTPS

Браузер разрешит `getUserMedia` только на **localhost** или по **HTTPS**. За **nginx** с TLS (см. `nginx-clinvoice.conf.example`) приложение должно открываться как `https://ваш-домен`.

Проверка: откройте страницу, включите запись и посмотрите блок «Накоплено под Whisper» — секунды должны расти.

## ICE / STUN / TURN

Если аудио не идёт (буфер ~0 с), часто виноват **обход NAT** или файрвол:

- У ВМ должен быть **исходящий** интернет (для STUN).
- У клиента не должны блокироваться **UDP** для ICE (иногда помогает только **TURN**).

В приложении можно задать JSON со списком ICE-серверов через переменную окружения или секрет Streamlit:

- `CLINVOICE_WEBRTC_ICE_SERVERS_JSON`

Пример (подставьте свои креды TURN при необходимости):

```json
[
  {"urls": "stun:stun.l.google.com:19302"},
  {"urls": "turn:turn.example.com:3478", "username": "u", "credential": "p"}
]
```

Логика сборки конфигурации: `resolve_webrtc_rtc_configuration()` в `app.py`.

## Логи браузера

При проблемах откройте DevTools → Console и вкладку, связанную с WebRTC (Chrome: `chrome://webrtc-internals`), чтобы увидеть состояние ICE и ошибки медиапотока.
