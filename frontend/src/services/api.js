import axios from 'axios'
const api = axios.create({baseURL:import.meta.env.VITE_API_URL || '',timeout:30000})
api.interceptors.response.use(r=>r, e=>Promise.reject(new Error(e.response?.data?.detail || e.message || 'API request failed')))
export const imageUrl = path => path?.startsWith('http') ? path : `${api.defaults.baseURL}${path || ''}`
export default api
