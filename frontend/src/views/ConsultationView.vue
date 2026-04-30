<script setup lang="ts">
import { ref, onBeforeUnmount, watch, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
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
        if (msg.type === 'speechkit_error' && msg.message) liveError.value = msg.message
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
  processor?.disconnect()
  sourceNode?.disconnect()
  processor = null
  sourceNode = null
  mediaStream?.getTracks().forEach((t) => t.stop())
  mediaStream = null
  void audioCtx?.close()
  audioCtx = null
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
  finalizing.value = true
  liveError.value = ''
  try {
    const r = await api.finalizeConsultation(consultationId.value)
    transcriptView.value = r.transcription
    protocolText.value = r.protocol_editor_text
    originalDone.value = true
    setStatus('Готово: текст и протокол обновлены.', 'ok')
  } catch (e: unknown) {
    setStatus(e instanceof Error ? e.message : String(e), 'err')
  } finally {
    finalizing.value = false
  }
}

function downloadText(filename: string, text: string) {
  const blob = new Blob([text], { type: 'text/plain;charset=utf-8' })
  const a = document.createElement('a')
  a.href = URL.createObjectURL(blob)
  a.download = filename
  a.click()
  URL.revokeObjectURL(a.href)
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
      <label class="row">
        <input v-model="recordPaused" type="checkbox" @change="togglePause" />
        Пауза
      </label>
      <div class="row">
        <button v-if="!recording" type="button" @click="startMic">Начать запись с микрофона</button>
        <button v-else type="button" class="danger" @click="stopMic">Остановить микрофон</button>
        <button type="button" :disabled="busy" @click="saveSnapshot">Сохранить черновик</button>
        <button type="button" :disabled="busy" @click="resetBuf">Сбросить буфер PCM</button>
      </div>
      <p v-if="statusMsg" :class="statusKind === 'ok' ? 'info' : 'err'">{{ statusMsg }}</p>
      <p v-if="liveError" class="err">{{ liveError }}</p>
    </section>

    <section class="card">
      <h2>Черновик</h2>
      <pre class="draft">{{ draft || '—' }}</pre>
    </section>

    <section class="card">
      <h2>Транскрипт</h2>
      <p v-if="!originalDone" class="muted small">После «Заполнить протокол» здесь появится уточнённый текст.</p>
      <pre class="draft">{{ transcriptView || '—' }}</pre>
    </section>

    <section class="card">
      <h2>Протокол</h2>
      <textarea v-model="protocolText" rows="14" class="proto" placeholder="Дата:, Жалобы:, …" />
      <div class="row">
        <button type="button" :disabled="finalizing || !consultationId" @click="doFinalize">
          {{ finalizing ? 'Обработка…' : 'Заполнить протокол' }}
        </button>
        <button type="button" :disabled="!protocolText" @click="downloadText('протокол.txt', protocolText)">
          Скачать протокол .txt
        </button>
      </div>
    </section>
  </div>
</template>

<style scoped>
.page {
  max-width: 880px;
  margin: 0 auto;
  padding: 1rem 1.25rem 3rem;
  font-family: system-ui, sans-serif;
  color: #000;
  background: #fff;
}
.head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
  flex-wrap: wrap;
  padding-bottom: 0.75rem;
  border-bottom: 1px solid #000;
}
h1 {
  margin: 0;
  font-size: 1.35rem;
  color: #000;
}
.actions {
  display: flex;
  gap: 0.75rem;
}
.muted {
  color: #000;
}
.small {
  font-size: 0.85rem;
  word-break: break-all;
}
.card {
  border: 1px solid #000;
  border-radius: 0;
  padding: 1rem 1.1rem;
  margin-top: 1rem;
  background: #fff;
}
h2 {
  margin: 0 0 0.75rem;
  font-size: 1.05rem;
  color: #000;
}
.row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  align-items: center;
  margin-top: 0.5rem;
  color: #000;
}
.row input[type='checkbox'] {
  width: 1rem;
  height: 1rem;
  accent-color: #000;
}
button {
  padding: 0.45rem 0.75rem;
  border-radius: 0;
  border: 1px solid #000;
  background: #fff;
  color: #000;
  cursor: pointer;
}
button.linkish {
  border: none;
  background: transparent;
  color: #1a5f7a;
  text-decoration: underline;
  padding: 0.25rem 0.35rem;
}
button.linkish:hover {
  color: #0d4a63;
}
button:disabled {
  opacity: 0.45;
  cursor: default;
}
button.danger {
  border-color: #000;
  color: #000;
}
.draft {
  white-space: pre-wrap;
  margin: 0;
  padding: 0.75rem;
  background: #fff;
  border: 1px solid #000;
  border-radius: 0;
  min-height: 3rem;
  font-size: 0.9rem;
  color: #000;
}
.proto {
  width: 100%;
  box-sizing: border-box;
  font-family: inherit;
  font-size: 0.9rem;
  padding: 0.5rem;
  border-radius: 0;
  border: 1px solid #000;
  background: #fff;
  color: #000;
}
.info {
  color: #15601d;
  margin: 0.5rem 0 0;
}
.err {
  color: #a40000;
  margin: 0.5rem 0 0;
}
code {
  font-size: 0.85rem;
  padding: 0.15rem 0.35rem;
  border: 1px solid #000;
  background: #fff;
  color: #000;
}
</style>
