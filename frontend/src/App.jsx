import { useState, useRef, useEffect, useCallback } from 'react'

function App() {
  const [file, setFile] = useState(null)
  const [data, setData] = useState(null)
  const [processing, setProcessing] = useState(false)
  const [percent, setPercent] = useState(0)
  const [error, setError] = useState(null)
  const pollRef = useRef(null)

  function stopPolling() {
    if (pollRef.current) {
      clearInterval(pollRef.current)
      pollRef.current = null
    }
  }

  const startPolling = useCallback(() => {
    stopPolling()
    pollRef.current = setInterval(async () => {
      const res = await fetch('http://localhost:8000/track')
      const job = await res.json()

      if (job.status === 'done') {
        stopPolling()
        setPercent(100)
        setData(job.result)
        setProcessing(false)
      } else if (job.status === 'error') {
        stopPolling()
        setError(job.message)
        setProcessing(false)
      } else {
        setPercent(job.percent ?? 0)
      }
    }, 1500)
  }, [])

  // On mount: if the backend is already processing (e.g. after a page refresh),
  // re-attach the polling loop so the progress bar picks up where it left off.
  useEffect(() => {
    fetch('http://localhost:8000/track')
      .then((r) => r.json())
      .then((job) => {
        if (job.status === 'processing') {
          setProcessing(true)
          setPercent(job.percent ?? 0)
          startPolling()
        }
      })
      .catch(() => {}) // backend not up yet — ignore

    return () => stopPolling()
  }, [startPolling])

  async function handleSubmit(e) {
    e.preventDefault()
    setProcessing(true)
    setData(null)
    setError(null)
    setPercent(0)

    const form = new FormData()
    form.append('video', file)
    await fetch('http://localhost:8000/track', { method: 'POST', body: form })

    startPolling()
  }

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <input type="file" accept="video/*" onChange={(e) => setFile(e.target.files[0])} />
        <button type="submit" disabled={!file || processing}>
          {processing ? 'Processing...' : 'Upload'}
        </button>
      </form>
      {processing && (
        <>
          <p>Processing video — this takes 30–60 seconds…</p>
          <progress value={percent} max={100} />
        </>
      )}
      {error && <p style={{ color: 'red' }}>Error: {error}</p>}
      {data && <video src={data.video_url} controls />}
      {data?.tracks?.length > 0 && (
        <table>
          <thead>
            <tr>
              <th>Track ID</th>
              <th>Label</th>
              <th>First Seen (s)</th>
              <th>Last Seen (s)</th>
              <th>Time on Screen (s)</th>
              <th>Distance (px)</th>
            </tr>
          </thead>
          <tbody>
            {data.tracks.map((t) => (
              <tr key={t.track_id}>
                <td>{t.track_id}</td>
                <td>{t.label}</td>
                <td>{t.first_seen_s}</td>
                <td>{t.last_seen_s}</td>
                <td>{t.time_on_screen_s}</td>
                <td>{t.distance_px}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

export default App
