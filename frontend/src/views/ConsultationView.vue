<script setup lang="ts">
import { ref, onBeforeUnmount, watch, computed } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import * as api from '../api'

const route = useRoute()
const router = useRouter()

const consultationId = ref('')
const creatingConsultation = ref(false)
const draft = ref('')
const speechkitErr = ref('')
const statusMsg = ref('')
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
        if (msg.type === 'speechkit_error' && msg.message) speechkitErr.value = msg.message
        if (msg.type === 'warn' && msg.message) statusMsg.value = msg.message
        if (msg.type === 'error' && msg.message) speechkitErr.value = msg.message
      } catch {
        /* ignore */
      }
    }
  }
  ws.onopen = () => {
    speechkitErr.value = ''
  }
  ws.onerror = () => {
    speechkitErr.value = 'Ошибка WebSocket'
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
  statusMsg.value = ''
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
    statusMsg.value = 'Черновик сохранён'
  } catch (e: unknown) {
    statusMsg.value = e instanceof Error ? e.message : String(e)
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
    statusMsg.value = 'Буфер PCM сброшен'
  } catch (e: unknown) {
    statusMsg.value = e instanceof Error ? e.message : String(e)
  } finally {
    busy.value = false
  }
}

async function doFinalize() {
  if (!consultationId.value) return
  finalizing.value = true
  speechkitErr.value = ''
  try {
    const r = await api.finalizeConsultation(consultationId.value)
    transcriptView.value = r.transcription
    protocolText.value = r.protocol_editor_text
    originalDone.value = true
    statusMsg.value = 'Готово: Whisper и протокол обновлены.'
  } catch (e: unknown) {
    statusMsg.value = e instanceof Error ? e.message : String(e)
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
      statusMsg.value = ''
    } catch {
      statusMsg.value = 'Не удалось загрузить консультацию'
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
        Пауза (не накапливать и не слать в SpeechKit)
      </label>
      <div class="row">
        <button v-if="!recording" type="button" @click="startMic">Начать запись с микрофона</button>
        <button v-else type="button" class="danger" @click="stopMic">Остановить микрофон</button>
        <button type="button" :disabled="busy" @click="saveSnapshot">Сохранить черновик</button>
        <button type="button" :disabled="busy" @click="resetBuf">Сбросить буфер PCM</button>
      </div>
      <p v-if="statusMsg" class="info">{{ statusMsg }}</p>
      <p v-if="speechkitErr" class="err">{{ speechkitErr }}</p>
    </section>

    <section class="card">
      <h2>Черновик (SpeechKit)</h2>
      <pre class="draft">{{ draft || '—' }}</pre>
    </section>

    <section class="card">
      <h2>Транскрипт</h2>
      <p v-if="!originalDone" class="muted small">После «Заполнить протокол» здесь появится уточнённый Whisper текст.</p>
      <pre class="draft">{{ transcriptView || '—' }}</pre>
    </section>

    <section class="card">
      <h2>Протокол</h2>
      <textarea v-model="protocolText" rows="14" class="proto" placeholder="Дата:, Жалобы:, …" />
      <div class="row">
        <button type="button" :disabled="finalizing || !consultationId" @click="doFinalize">
          {{ finalizing ? 'Обработка…' : 'Заполнить протокол (Whisper + YandexGPT)' }}
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
}
.head {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 1rem;
}
h1 {
  margin: 0;
  font-size: 1.35rem;
}
.actions {
  display: flex;
  gap: 0.75rem;
}
.linkish {
  background: none;
  border: none;
  color: #1a5f7a;
  cursor: pointer;
  text-decoration: underline;
  padding: 0.25rem;
}
.muted {
  color: #555;
}
.small {
  font-size: 0.85rem;
  word-break: break-all;
}
.card {
  border: 1px solid #ddd;
  border-radius: 8px;
  padding: 1rem 1.1rem;
  margin-top: 1rem;
  background: #fafafa;
}
h2 {
  margin: 0 0 0.75rem;
  font-size: 1.05rem;
}
.row {
  display: flex;
  flex-wrap: wrap;
  gap: 0.5rem;
  align-items: center;
  margin-top: 0.5rem;
}
button {
  padding: 0.45rem 0.75rem;
  border-radius: 6px;
  border: 1px solid #ccc;
  background: #fff;
  cursor: pointer;
}
button.danger {
  border-color: #c44;
  color: #a22;
}
.draft {
  white-space: pre-wrap;
  margin: 0;
  padding: 0.75rem;
  background: #fff;
  border: 1px solid #e0e0e0;
  border-radius: 6px;
  min-height: 3rem;
  font-size: 0.9rem;
}
.proto {
  width: 100%;
  box-sizing: border-box;
  font-family: inherit;
  font-size: 0.9rem;
  padding: 0.5rem;
  border-radius: 6px;
  border: 1px solid #ccc;
}
.info {
  color: #0a5;
  margin: 0.5rem 0 0;
}
.err {
  color: #a00;
  margin: 0.5rem 0 0;
}
code {
  font-size: 0.85rem;
}
</style>
