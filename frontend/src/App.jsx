import { useState, useRef } from 'react'

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

  async function handleSubmit(e) {
    e.preventDefault()
    setProcessing(true)
    setData(null)
    setError(null)
    setPercent(0)

    const form = new FormData()
    form.append('video', file)
    await fetch('http://localhost:8000/track', { method: 'POST', body: form })

    // Poll GET /track every 1.5 s until done or error
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
              <th>Time on Screen (s)</th>
            </tr>
          </thead>
          <tbody>
            {data.tracks.map((t) => (
              <tr key={t.track_id}>
                <td>{t.track_id}</td>
                <td>{t.label}</td>
                <td>{t.time_on_screen_s}</td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  )
}

export default App
