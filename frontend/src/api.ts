import axios from 'axios'

export const http = axios.create({
  baseURL: '',
  withCredentials: true,
})

export type Snapshot = {
  live_transcript_editor: string
  live_transcript_pause_auto_sync: boolean
  original_transcription: string | null
  doctor_transcript_editor: string
  protocol_editor_text: string
  protocol_consultation_date: string
  status: string
}

export async function login(username: string, password: string) {
  await http.post('/api/auth/login', { username, password })
}

export async function logout() {
  await http.post('/api/auth/logout')
}

export async function me() {
  const { data } = await http.get<{ id: number; username: string }>('/api/auth/me')
  return data
}

export async function createConsultation() {
  const { data } = await http.post<{ id: string }>('/api/consultations')
  return data.id
}

export async function getConsultation(id: string) {
  const { data } = await http.get<{
    id: string
    created_at: number
    updated_at: number
    status: string
    snapshot: Snapshot | null
  }>(`/api/consultations/${encodeURIComponent(id)}`)
  return data
}

export async function getSnapshot(id: string) {
  const { data } = await http.get<{ snapshot: Snapshot }>(
    `/api/consultations/${encodeURIComponent(id)}/snapshot`,
  )
  return data.snapshot
}

export async function putSnapshot(id: string, snapshot: Partial<Snapshot>) {
  const { data } = await http.put<{ snapshot: Snapshot }>(
    `/api/consultations/${encodeURIComponent(id)}/snapshot`,
    snapshot,
  )
  return data.snapshot
}

export async function resetAudio(id: string) {
  await http.post(`/api/consultations/${encodeURIComponent(id)}/reset-audio`)
}

export async function finalizeConsultation(id: string) {
  const { data } = await http.post<{
    transcription: string
    protocol: Record<string, string>
    protocol_editor_text: string
    protocol_consultation_date: string
    transcript_txt: string
    protocol_txt: string
  }>(`/api/consultations/${encodeURIComponent(id)}/finalize`)
  return data
}
