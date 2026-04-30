<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { login, me } from '../api'

const route = useRoute()
const router = useRouter()
const username = ref('')
const password = ref('')
const showPassword = ref(false)
const error = ref('')
const busy = ref(false)

onMounted(async () => {
  try {
    await me()
    const r = (route.query.redirect as string) || '/c/new'
    await router.replace(r)
  } catch {
    /* not logged in */
  }
})

async function submit() {
  error.value = ''
  busy.value = true
  try {
    await login(username.value.trim(), password.value)
    const r = (route.query.redirect as string) || '/c/new'
    await router.replace(r)
  } catch (e: unknown) {
    error.value = e instanceof Error ? e.message : 'Ошибка входа'
  } finally {
    busy.value = false
  }
}
</script>

<template>
  <div class="page">
    <h1>ClinVoice</h1>
    <p class="muted">Вход</p>
    <form class="card" @submit.prevent="submit">
      <label>Имя пользователя</label>
      <input v-model="username" type="text" autocomplete="username" required />
      <label>Пароль</label>
      <div class="password-wrap">
        <input
          v-model="password"
          :type="showPassword ? 'text' : 'password'"
          autocomplete="current-password"
          required
        />
        <button
          type="button"
          class="pw-toggle"
          :aria-pressed="showPassword"
          :aria-label="showPassword ? 'Скрыть пароль' : 'Показать пароль'"
          :title="showPassword ? 'Скрыть пароль' : 'Показать пароль'"
          @click="showPassword = !showPassword"
        >
          <svg
            v-if="!showPassword"
            xmlns="http://www.w3.org/2000/svg"
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
            aria-hidden="true"
          >
            <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
            <circle cx="12" cy="12" r="3" />
          </svg>
          <svg
            v-else
            xmlns="http://www.w3.org/2000/svg"
            width="20"
            height="20"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            stroke-width="2"
            stroke-linecap="round"
            stroke-linejoin="round"
            aria-hidden="true"
          >
            <path
              d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19m-6.72-1.07a3 3 0 1 1-4.24-4.24"
            />
            <line x1="1" y1="1" x2="23" y2="23" />
          </svg>
        </button>
      </div>
      <p v-if="error" class="err">{{ error }}</p>
      <button type="submit" :disabled="busy">{{ busy ? '…' : 'Войти' }}</button>
    </form>
  </div>
</template>

<style scoped>
.page {
  max-width: 420px;
  margin: 2rem auto 0;
  padding: 0 0.25rem;
}
.muted {
  margin: 0 0 1.25rem;
  color: var(--text);
}
.card {
  display: flex;
  flex-direction: column;
  gap: 0.55rem;
  padding: 1.1rem;
  border: 1px solid var(--border);
  border-radius: var(--clinvoice-radius);
  background: var(--bg);
  box-shadow: var(--shadow);
}
label {
  font-size: 0.85rem;
  font-weight: 600;
  color: var(--text-h);
}
.password-wrap {
  position: relative;
}
.password-wrap input {
  width: 100%;
  box-sizing: border-box;
  padding-right: 2.75rem;
}
.password-wrap .pw-toggle {
  position: absolute;
  right: 0.25rem;
  top: 50%;
  transform: translateY(-50%);
}
.card > button[type='submit'] {
  margin-top: 0.35rem;
}
.err {
  color: #b91c1c;
  margin: 0;
}
@media (prefers-color-scheme: dark) {
  .err {
    color: #fca5a5;
  }
}
</style>
