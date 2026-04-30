import { createRouter, createWebHistory } from 'vue-router'
import { me } from '../api'
import LoginView from '../views/LoginView.vue'
import ConsultationView from '../views/ConsultationView.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    { path: '/', redirect: '/login' },
    { path: '/login', name: 'login', component: LoginView },
    { path: '/c/:id', name: 'consultation', component: ConsultationView, meta: { requiresAuth: true } },
  ],
})

router.beforeEach(async (to) => {
  if (!to.meta.requiresAuth) return true
  try {
    await me()
    return true
  } catch {
    return { name: 'login', query: { redirect: to.fullPath } }
  }
})

export default router
