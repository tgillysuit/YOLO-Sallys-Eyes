import { useState } from 'react'

function App() {
  const [file, setFile] = useState(null)
  const [data, setData] = useState(null)
  const [processing, setProcessing] = useState(false)

  async function handleSubmit(e) {
    e.preventDefault()
    setProcessing(true)
    setData(null)
    try {
      const form = new FormData()
      form.append('video', file)
      const res = await fetch('http://localhost:8000/track', { method: 'POST', body: form })
      setData(await res.json())
    } finally {
      setProcessing(false)
    }
  }

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <input type="file" accept="video/*" onChange={(e) => setFile(e.target.files[0])} />
        <button type="submit" disabled={!file || processing}>
          {processing ? 'Processing...' : 'Upload'}
        </button>
      </form>
      {processing && <p>Processing video — this takes 30–60 seconds…</p>}
      {data && <video src={data.video_url} controls />}
    </div>
  )
}

export default App
