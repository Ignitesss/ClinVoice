<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useRoute, useRouter } from 'vue-router'
import { login, me } from '../api'

const route = useRoute()
const router = useRouter()
const username = ref('')
const password = ref('')
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
      <input v-model="password" type="password" autocomplete="current-password" required />
      <p v-if="error" class="err">{{ error }}</p>
      <button type="submit" :disabled="busy">{{ busy ? '…' : 'Войти' }}</button>
    </form>
  </div>
</template>

<style scoped>
.page {
  max-width: 420px;
  margin: 3rem auto;
  padding: 1rem;
  color: #000;
  background: #fff;
}
h1 {
  margin: 0 0 0.25rem;
  color: #000;
}
.muted {
  color: #000;
  margin: 0 0 1.5rem;
}
.card {
  display: flex;
  flex-direction: column;
  gap: 0.5rem;
  padding: 1rem;
  border: 1px solid #000;
  background: #fff;
}
label {
  font-size: 0.85rem;
  font-weight: 600;
  color: #000;
}
input {
  padding: 0.5rem 0.6rem;
  border: 1px solid #000;
  border-radius: 0;
  background: #fff;
  color: #000;
}
button {
  margin-top: 0.5rem;
  padding: 0.55rem 1rem;
  border: 1px solid #000;
  border-radius: 0;
  background: #fff;
  color: #000;
  cursor: pointer;
}
button:disabled {
  opacity: 0.45;
  cursor: default;
}
.err {
  color: #000;
  margin: 0;
}
</style>
