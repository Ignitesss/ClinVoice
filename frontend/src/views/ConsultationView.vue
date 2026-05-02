<script setup lang="ts">
import { ref, onBeforeUnmount, watch, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { isAxiosError } from 'axios'
import * as api from '../api'

const route = useRoute()
const router = useRouter()

const consultationId = ref('')
const creatingConsultation = ref(false)
const draft = ref('')
const liveError = ref('')
const statusMsg = ref('')
/** цвет строки statusMsg: зелёный (ok) или красный (err) */
const statusKind = ref<'ok' | 'err'>('ok')
const busy = ref(false)
const finalizing = ref(false)

const protocolText = ref('')
const transcriptView = ref('')
const originalDone = ref(false)

const recording = ref(false)
const recordPaused = ref(false)

let ws: WebSocket | null = null
let audioCtx: AudioContext | null = null
let mediaStream: MediaStream | null = null
let processor: ScriptProcessorNode | null = null
let sourceNode: MediaStreamAudioSourceNode | null = null

const OUT_SR = 16000

function setStatus(msg: string, kind: 'ok' | 'err') {
  statusKind.value = kind
  statusMsg.value = msg
}

function clearStatus() {
  statusMsg.value = ''
}

/** 524/504: иногда нет e.response (обрыв), но в тексте есть код. */
function finalizeErrorLooksLikeProxyTimeout(e: unknown): boolean {
  if (isAxiosError(e)) {
    const s = e.response?.status
    if (s === 524 || s === 504) return true
    const m = (e.message || '').toLowerCase()
    if (m.includes('524') || m.includes('504')) return true
  }
  const msg = (e instanceof Error ? e.message : String(e)).toLowerCase()
  return /\b524\b|\b504\b/.test(msg)
}

/**
 * После таймаута прокси сервер может ещё секунды обрабатывать finalize — первая GET сразу часто пустая.
 * Несколько попыток с паузой вместо одной мгновенной загрузки.
 */
async function recoverSnapshotFromServer(id: string): Promise<boolean> {
  const firstDelayMs = 2500
  const stepMs = 3500
  const maxSteps = 14
  await new Promise<void>((r) => setTimeout(r, firstDelayMs))
  for (let step = 0; step < maxSteps; step++) {
    if (step > 0) {
      await new Promise<void>((r) => setTimeout(r, stepMs))
    }
    setStatus(
      `Прокси оборвал ответ, но сервер мог доработать — подгружаем черновик (попытка ${step + 1}/${maxSteps})…`,
      'ok',
    )
    try {
      await loadConsultation(id)
    } catch {
      continue
    }
    if (transcriptView.value.trim() || protocolText.value.trim()) {
      originalDone.value = Boolean(transcriptView.value.trim())
      return true
    }
  }
  return false
}

const reloadSnapshotBusy = ref(false)

async function reloadFromServer() {
  if (!consultationId.value) return
  reloadSnapshotBusy.value = true
  try {
    await loadConsultation(consultationId.value)
    if (transcriptView.value.trim() || protocolText.value.trim()) {
      originalDone.value = Boolean(transcriptView.value.trim())
      setStatus('Данные с сервера обновлены.', 'ok')
    } else {
      setStatus('Для этой консультации на сервере пока пустой черновик (транскрипт/протокол).', 'ok')
    }
  } catch (e: unknown) {
    setStatus(e instanceof Error ? e.message : String(e), 'err')
  } finally {
    reloadSnapshotBusy.value = false
  }
}

function floatToInt16Downsample(input: Float32Array, inRate: number, outRate: number): ArrayBuffer {
  if (inRate === outRate) {
    const out = new Int16Array(input.length)
    for (let i = 0; i < input.length; i++) {
      const s = Math.max(-1, Math.min(1, input[i]!))
      out[i] = s < 0 ? (s * 0x8000) | 0 : (s * 0x7fff) | 0
    }
    return out.buffer
  }
  const ratio = inRate / outRate
  const outLen = Math.floor(input.length / ratio)
  const out = new Int16Array(outLen)
  for (let i = 0; i < outLen; i++) {
    const src = Math.min(input.length - 1, Math.floor(i * ratio))
    const s = Math.max(-1, Math.min(1, input[src]!))
    out[i] = s < 0 ? (s * 0x8000) | 0 : (s * 0x7fff) | 0
  }
  return out.buffer
}

function connectWs() {
  if (!consultationId.value) return
  ws?.close()
  const proto = location.protocol === 'https:' ? 'wss:' : 'ws:'
  const url = `${proto}//${location.host}/ws/audio?consultation_id=${encodeURIComponent(consultationId.value)}`
  ws = new WebSocket(url)
  ws.binaryType = 'arraybuffer'
  ws.onmessage = (ev) => {
    if (typeof ev.data === 'string') {
      try {
        const msg = JSON.parse(ev.data) as { type: string; text?: string; message?: string }
        if (msg.type === 'draft' && msg.text !== undefined) draft.value = msg.text
        if ((msg.type === 'draft_error' || msg.type === 'speechkit_error') && msg.message)
          liveError.value = msg.message
        if (msg.type === 'warn' && msg.message) setStatus(msg.message, 'ok')
        if (msg.type === 'error' && msg.message) liveError.value = msg.message
      } catch {
        /* ignore */
      }
    }
  }
  ws.onopen = () => {
    liveError.value = ''
  }
  ws.onerror = () => {
    liveError.value = 'Ошибка WebSocket'
  }
}

function sendPause() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ type: 'pause', value: recordPaused.value }))
  }
}

async function loadConsultation(id: string) {
  const data = await api.getConsultation(id)
  const snap = data.snapshot
  if (snap) {
    protocolText.value = snap.protocol_editor_text || ''
    transcriptView.value =
      snap.doctor_transcript_editor || snap.live_transcript_editor || snap.original_transcription || ''
    originalDone.value = Boolean(snap.original_transcription)
  } else {
    protocolText.value = ''
    transcriptView.value = ''
    originalDone.value = false
  }
}

async function startMic() {
  if (recording.value) return
  clearStatus()
  recordPaused.value = false
  mediaStream = await navigator.mediaDevices.getUserMedia({ audio: true, video: false })
  audioCtx = new AudioContext()
  const inRate = audioCtx.sampleRate
  sourceNode = audioCtx.createMediaStreamSource(mediaStream)
  processor = audioCtx.createScriptProcessor(4096, 1, 1)
  processor.onaudioprocess = (e) => {
    if (!recording.value || recordPaused.value || !ws || ws.readyState !== WebSocket.OPEN) return
    const input = e.inputBuffer.getChannelData(0)
    const buf = floatToInt16Downsample(input, inRate, OUT_SR)
    ws.send(buf)
  }
  const gain = audioCtx.createGain()
  gain.gain.value = 0
  sourceNode.connect(processor)
  processor.connect(gain)
  gain.connect(audioCtx.destination)
  recording.value = true
  sendPause()
}

function stopMic() {
  recording.value = false
  recordPaused.value = false
  processor?.disconnect()
  sourceNode?.disconnect()
  processor = null
  sourceNode = null
  mediaStream?.getTracks().forEach((t) => t.stop())
  mediaStream = null
  void audioCtx?.close()
  audioCtx = null
  sendPause()
}

async function saveSnapshot() {
  if (!consultationId.value) return
  busy.value = true
  try {
    await api.putSnapshot(consultationId.value, {
      protocol_editor_text: protocolText.value,
      doctor_transcript_editor: transcriptView.value,
      live_transcript_editor: transcriptView.value,
    })
    setStatus('Черновик сохранён', 'ok')
  } catch (e: unknown) {
    setStatus(e instanceof Error ? e.message : String(e), 'err')
  } finally {
    busy.value = false
  }
}

async function resetBuf() {
  if (!consultationId.value) return
  busy.value = true
  try {
    await api.resetAudio(consultationId.value)
    draft.value = ''
    setStatus('Буфер PCM сброшен', 'ok')
  } catch (e: unknown) {
    setStatus(e instanceof Error ? e.message : String(e), 'err')
  } finally {
    busy.value = false
  }
}

async function doFinalize() {
  if (!consultationId.value) return
  stopMic()
  finalizing.value = true
  liveError.value = ''
  try {
    const r = await api.finalizeConsultation(consultationId.value)
    transcriptView.value = r.transcription
    protocolText.value = r.protocol_editor_text
    originalDone.value = true
    setStatus('Готово: текст и протокол обновлены.', 'ok')
    downloadText('транскрипт.txt', r.transcript_txt)
    window.setTimeout(() => {
      downloadText('протокол.txt', r.protocol_txt)
    }, 500)
  } catch (e: unknown) {
    if (finalizeErrorLooksLikeProxyTimeout(e)) {
      setStatus(
        'Таймаут прокси (524/504): ответ не дошёл, но сервер может ещё писать черновик — подождём и несколько раз запросим снимок…',
        'ok',
      )
      const recovered = await recoverSnapshotFromServer(consultationId.value)
      if (recovered) {
        setStatus(
          'Черновик подгружен с сервера (транскрипт/протокол). При таймауте прокси обработка часто заканчивается позже первого ответа.',
          'ok',
        )
      } else {
        setStatus(
          'Таймаут прокси: черновик на сервере так и не появился за отведённое время. Обновите страницу позже или нажмите «Подгрузить с сервера». Долгий ответ — см. deploy/CLOUDFLARE_TUNNEL.md.',
          'err',
        )
      }
    } else {
      setStatus(e instanceof Error ? e.message : String(e), 'err')
    }
  } finally {
    finalizing.value = false
  }
}

function downloadText(filename: string, text: string) {
  const blob = new Blob([text], { type: 'text/plain;charset=utf-8' })
  const a = document.createElement('a')
  const url = URL.createObjectURL(blob)
  a.href = url
  a.download = filename
  a.rel = 'noopener'
  a.style.display = 'none'
  document.body.appendChild(a)
  a.click()
  window.setTimeout(() => {
    a.remove()
    URL.revokeObjectURL(url)
  }, 1500)
}

async function newConsultation() {
  stopMic()
  ws?.close()
  await router.push('/c/new')
}

async function logout() {
  stopMic()
  ws?.close()
  await api.logout()
  await router.push('/login')
}

const shareUrl = computed(() => {
  if (!consultationId.value) return ''
  const u = new URL(location.href)
  u.pathname = `/c/${consultationId.value}`
  u.search = ''
  return u.toString()
})

watch(
  () => route.params.id,
  async (pid) => {
    let id = pid as string | undefined
    if (!id) return
    if (id === 'new') {
      if (creatingConsultation.value) return
      creatingConsultation.value = true
      try {
        id = await api.createConsultation()
        await router.replace(`/c/${id}`)
        consultationId.value = id
      } finally {
        creatingConsultation.value = false
      }
    } else {
      consultationId.value = id
    }
    stopMic()
    ws?.close()
    try {
      await loadConsultation(consultationId.value)
      clearStatus()
    } catch {
      setStatus('Не удалось загрузить консультацию', 'err')
    }
    connectWs()
  },
  { immediate: true },
)

onBeforeUnmount(() => {
  stopMic()
  ws?.close()
})

function togglePause() {
  sendPause()
}
</script>

<template>
  <div class="page">
    <header class="head">
      <h1>ClinVoice</h1>
      <div class="actions">
        <button type="button" class="linkish" @click="newConsultation">Новая консультация</button>
        <button type="button" class="linkish" @click="logout">Выход</button>
      </div>
    </header>

    <p class="muted">Идентификатор: <code>{{ consultationId }}</code></p>
    <p class="muted small">Ссылка для возврата: <a :href="shareUrl">{{ shareUrl }}</a></p>

    <section class="card">
      <h2>Запись</h2>
      <p class="hint">
        <strong>Пауза</strong> — при включённой записи не отправлять звук на сервер (микрофон остаётся включённым).
        <strong>Остановить запись</strong> — полностью выключить микрофон.
      </p>
      <label class="row pause-row" :class="{ disabled: !recording }">
        <input
          v-model="recordPaused"
          type="checkbox"
          :disabled="!recording"
          @change="togglePause"
        />
        Пауза
      </label>
      <div class="row">
        <button v-if="!recording" type="button" @click="startMic">Начать запись</button>
        <button v-else type="button" class="danger" @click="stopMic">Остановить запись</button>
        <button type="button" :disabled="busy" @click="saveSnapshot">Сохранить черновик</button>
        <button type="button" :disabled="reloadSnapshotBusy || !consultationId" @click="reloadFromServer">
          Подгрузить с сервера
        </button>
        <button type="button" :disabled="busy" @click="resetBuf">Сбросить буфер PCM</button>
      </div>
      <p v-if="statusMsg" :class="statusKind === 'ok' ? 'info' : 'err'">{{ statusMsg }}</p>
      <p v-if="liveError" class="err">{{ liveError }}</p>
    </section>

    <section class="card">
      <div class="draft-head">
        <h2>Черновик</h2>
        <button type="button" :disabled="finalizing || !consultationId" class="finalize-btn" @click="doFinalize">
          {{ finalizing ? 'Обработка…' : 'Заполнить протокол' }}
        </button>
      </div>
      <pre class="draft">{{ draft || '—' }}</pre>
    </section>

    <section class="card">
      <h2>Транскрипт</h2>
      <p v-if="!originalDone" class="muted small">После «Заполнить протокол» здесь появится уточнённый текст.</p>
      <pre class="draft">{{ transcriptView || '—' }}</pre>
      <div class="row">
        <button
          type="button"
          :disabled="!transcriptView.trim()"
          @click="downloadText('транскрипт.txt', transcriptView)"
        >
          Скачать транскрипт .txt
        </button>
      </div>
    </section>

    <section class="card">
      <h2>Протокол</h2>
      <textarea v-model="protocolText" rows="14" class="proto" placeholder="Дата:, Жалобы:, …" />
      <div class="row">
        <button type="button" :disabled="!protocolText" @click="downloadText('протокол.txt', protocolText)">
          Скачать протокол .txt
        </button>
      </div>
    </section>
  </div>
</template>

<style scoped>
.page {
  width: 100%;
  padding: 0 0.15rem 2rem;
}
.head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  flex-wrap: wrap;
  padding-bottom: 0.85rem;
  margin-bottom: 0.25rem;
  border-bottom: 1px solid var(--border);
}
.actions {
  display: flex;
  gap: 0.65rem;
  flex-wrap: wrap;
}
.muted {
  color: var(--text);
}
.small {
  font-size: 0.9rem;
  word-break: break-all;
}
.card {
  border: 1px solid var(--border);
  border-radius: var(--clinvoice-radius);
  padding: 1.1rem 1.15rem;
  margin-top: 1rem;
  background: var(--bg);
  box-shadow: var(--shadow);
}
.hint {
  font-size: 0.88rem;
  margin: 0 0 0.75rem;
  line-height: 1.45;
  color: var(--text);
}
.hint strong {
  font-weight: 600;
  color: var(--text-h);
}
.draft-head {
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  justify-content: space-between;
  gap: 0.5rem 1rem;
  margin-bottom: 0.75rem;
}
.draft-head h2 {
  margin: 0;
}
.pause-row.disabled {
  opacity: 0.45;
}
.row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  align-items: center;
  margin-top: 0.5rem;
  color: var(--text);
}
.draft {
  white-space: pre-wrap;
  margin: 0;
  padding: 0.85rem;
  background: var(--code-bg);
  border: 1px solid var(--border);
  border-radius: var(--clinvoice-radius);
  min-height: 3rem;
  font-size: 0.92rem;
  color: var(--text-h);
  font-family: var(--mono, ui-monospace, monospace);
}
.proto {
  width: 100%;
  box-sizing: border-box;
  margin-top: 0.35rem;
}
</style>
