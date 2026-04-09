import axios from 'axios'

const api = axios.create({
  baseURL: 'http://localhost:8000',
})

export function uploadAudio(file) {
  const form = new FormData()
  form.append('file', file)
  return api.post('/predict', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
  })
}

export function analyzeRecording(base64, format) {
  return api.post('/predict-base64', {
    audio: base64,
    format,
  })
}

export default api
