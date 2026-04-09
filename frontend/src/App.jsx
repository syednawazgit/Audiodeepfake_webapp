import { useCallback, useEffect, useRef, useState } from 'react'
import { useDropzone } from 'react-dropzone'
import { FiMic, FiUploadCloud } from 'react-icons/fi'
import { analyzeRecording, uploadAudio } from './api'
import './App.css'

const ACCEPT = { 'audio/wav': ['.wav'], 'audio/mpeg': ['.mp3'], 'audio/flac': ['.flac'], 'audio/ogg': ['.ogg'] }

function formatTime(totalSec) {
  const m = Math.floor(totalSec / 60)
  const s = totalSec % 60
  return `${String(m).padStart(2, '0')}:${String(s).padStart(2, '0')}`
}

function blobToBase64(blob) {
  return new Promise((resolve, reject) => {
    const r = new FileReader()
    r.onloadend = () => {
      const res = r.result
      if (typeof res !== 'string') {
        reject(new Error('read failed'))
        return
      }
      const i = res.indexOf(',')
      resolve(i >= 0 ? res.slice(i + 1) : res)
    }
    r.onerror = () => reject(r.error)
    r.readAsDataURL(blob)
  })
}

function errMessage(err) {
  const d = err?.response?.data?.detail
  if (typeof d === 'string') return d
  if (Array.isArray(d)) return d.map((x) => x.msg || String(x)).join(' ') || err.message
  if (d && typeof d === 'object') return JSON.stringify(d)
  return err?.message || 'Something went wrong'
}

export default function App() {
  const [file, setFile] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)
  const [result, setResult] = useState(null)

  const [recording, setRecording] = useState(false)
  const [recordSec, setRecordSec] = useState(0)
  const mediaRecorderRef = useRef(null)
  const chunksRef = useRef([])
  const streamRef = useRef(null)
  const timerRef = useRef(null)

  const onDrop = useCallback((accepted) => {
    setError(null)
    setResult(null)
    if (accepted?.[0]) setFile(accepted[0])
  }, [])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: ACCEPT,
    multiple: false,
    onDropRejected: () => setError('Only .wav, .mp3, .flac, and .ogg files are allowed.'),
  })

  const stopTimer = () => {
    if (timerRef.current) {
      clearInterval(timerRef.current)
      timerRef.current = null
    }
  }

  useEffect(() => {
    return () => {
      stopTimer()
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop())
        streamRef.current = null
      }
    }
  }, [])

  const startRecording = async () => {
    setError(null)
    setResult(null)
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
      const mime = MediaRecorder.isTypeSupported('audio/webm;codecs=opus')
        ? 'audio/webm;codecs=opus'
        : 'audio/webm'
      const mr = new MediaRecorder(stream, { mimeType: mime })
      mediaRecorderRef.current = mr
      chunksRef.current = []
      mr.ondataavailable = (e) => {
        if (e.data.size) chunksRef.current.push(e.data)
      }
      mr.start()
      setRecording(true)
      setRecordSec(0)
      stopTimer()
      timerRef.current = setInterval(() => setRecordSec((s) => s + 1), 1000)
    } catch (e) {
      setError(errMessage(e) || 'Microphone access denied or unavailable.')
    }
  }

  const stopRecordingAndAnalyze = async () => {
    const mr = mediaRecorderRef.current
    if (!mr || mr.state === 'inactive') {
      setRecording(false)
      stopTimer()
      return
    }

    return new Promise((resolve) => {
      mr.onstop = async () => {
        stopTimer()
        setRecording(false)
        if (streamRef.current) {
          streamRef.current.getTracks().forEach((t) => t.stop())
          streamRef.current = null
        }
        const blob = new Blob(chunksRef.current, { type: mr.mimeType || 'audio/webm' })
        chunksRef.current = []
        mediaRecorderRef.current = null
        if (blob.size < 100) {
          setError('Recording too short or empty.')
          resolve()
          return
        }
        setLoading(true)
        setError(null)
        try {
          const b64 = await blobToBase64(blob)
          const { data } = await analyzeRecording(b64, 'webm')
          setResult(data)
        } catch (e) {
          setError(errMessage(e))
        } finally {
          setLoading(false)
        }
        resolve()
      }
      mr.stop()
    })
  }

  const handleAnalyzeUpload = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    setResult(null)
    try {
      const { data } = await uploadAudio(file)
      setResult(data)
    } catch (e) {
      setError(errMessage(e))
    } finally {
      setLoading(false)
    }
  }

  const resetAll = () => {
    setFile(null)
    setResult(null)
    setError(null)
    setRecordSec(0)
  }

  const isFake = result?.prediction === 'Fake'

  return (
    <div className="app">
      {loading && (
        <div className="loading-overlay" role="status">
          <div className="spinner" />
          <p>Analyzing audio...</p>
        </div>
      )}

      {error && (
        <div className="toast" role="alert">
          {error}
        </div>
      )}

      <header className="header">
        <h1>
          Audio <span className="accent">Deepfake</span> Detector
        </h1>
        <p>
          Two-stream fusion model trained on <strong>ASVspoof 2019</strong> LA (LFCC CNN + Wav2Vec2-base).
          Upload or record audio for a <strong>spoof</strong> vs <strong>bonafide</strong> label.
        </p>
      </header>

      <div className="panels">
        <section className="card">
          <h2>
            <FiUploadCloud aria-hidden /> Upload audio
          </h2>
          <div {...getRootProps()} className={`dropzone ${isDragActive ? 'active' : ''}`}>
            <input {...getInputProps()} />
            <strong>Drag &amp; drop</strong>
            <p>or click to choose a file</p>
            <p className="hint">.wav, .mp3, .flac, .ogg</p>
          </div>
          {file && (
            <div className="file-meta">
              {file.name} — {(file.size / 1024).toFixed(1)} KB
            </div>
          )}
          <button type="button" className="btn btn-primary" disabled={!file || loading} onClick={handleAnalyzeUpload}>
            Analyze
          </button>
        </section>

        <section className="card">
          <h2>
            <FiMic aria-hidden /> Record audio
          </h2>
          {!recording && <p style={{ margin: 0, color: '#9898a8', fontSize: '0.9rem' }}>Uses your microphone; saved as WebM and sent to the API.</p>}
          <div className="record-timer">{formatTime(recordSec)}</div>
          {recording && (
            <div className="waveform" aria-hidden>
              {[1, 2, 3, 4, 5, 6, 7, 8].map((i) => (
                <span key={i} />
              ))}
            </div>
          )}
          {!recording ? (
            <button type="button" className="btn btn-secondary" disabled={loading} onClick={startRecording}>
              Start recording
            </button>
          ) : (
            <button type="button" className="btn btn-danger" onClick={() => void stopRecordingAndAnalyze()}>
              Stop &amp; analyze
            </button>
          )}
        </section>
      </div>

      {result && (
        <section className="results">
          <div className="results-inner results-inner--verdict-only">
            <div className={`verdict ${isFake ? 'fake' : 'real'}`}>
              {isFake ? 'Spoof / Fake' : 'Bonafide / Real'}
            </div>
            <p className="meta-small">Processing time: {result.processing_time_ms} ms</p>
          </div>
          <p className="disclaimer">Model trained on ASVspoof 2019 LA dataset. This label is an estimate, not legal proof.</p>
          <div className="reset-row">
            <button type="button" className="btn btn-secondary" onClick={resetAll}>
              Analyze another file
            </button>
          </div>
        </section>
      )}
    </div>
  )
}
