# ClinVoice через Cloudflare Tunnel

HTTPS до пользователя обеспечивает Cloudflare; на ВМ достаточно исходящего интернета и локального Streamlit на `127.0.0.1`.

## Быстрый тест без своего домена (Quick Tunnel)

На ВМ, рядом с запущенным Streamlit:

```bash
cloudflared tunnel --url http://127.0.0.1:8501
```

В логе появится URL вида `https://xxxxx.trycloudflare.com` — им можно пользоваться с чужих ПК. Ссылка при перезапуске обычно меняется.

## Постоянный адрес со своим доменом (Named Tunnel)

1. Аккаунт [Cloudflare](https://www.cloudflare.com/), домен добавлен в Cloudflare (DNS с оранжевым облаком по желанию для записей туннеля).
2. Zero Trust → **Networks** → **Tunnels** → **Create a tunnel** → установите `cloudflared` по инструкции, сохраните token/credentials.
3. В туннеле добавьте **Public Hostname**: ваш поддомен (например `clinvoice.example.com`) → **HTTP** → `localhost:8501` (или `127.0.0.1:8501`).
4. В DNS для этого же аккаунта Cloudflare создайте запись типа **CNAME** на `<tunnel-id>.cfargotunnel.com` (интерфейс часто подставляет её автоматически при сохранении hostname).

Пример локального файла конфига: [`cloudflared.config.example.yml`](cloudflared.config.example.yml) (для `cloudflared tunnel run` с флагом `--config`).

## Streamlit на ВМ

Слушать только loopback, чтобы порт не был открыт напрямую в интернет:

```bash
streamlit run app.py --server.address 127.0.0.1 --server.port 8501
```

Автозапуск после перезагрузки ВМ — пример unit-файла **systemd**: [`clinvoice-streamlit.service.example`](clinvoice-streamlit.service.example) (скопировать в `/etc/systemd/system/`, `daemon-reload`, `enable --now`). Отдельно так же можно оформить **`cloudflared`** вторым сервисом, если не используете встроенный запуск из инсталлятора туннеля.

## Примечания

- Порты **80/443 на ВМ открывать не нужно** — туннель инициируется **изнутри** ВМ наружу.
- Доступ по-прежнему защищается логином/паролем в ClinVoice; туннель лишь доставляет трафик по HTTPS.
