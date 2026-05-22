import { useState } from 'react'

function App() {
  const [file, setFile] = useState(null)
  const [data, setData] = useState(null)

  async function handleSubmit(e) {
    e.preventDefault()
    const form = new FormData()
    form.append('video', file)
    const res = await fetch('http://localhost:8000/track', { method: 'POST', body: form })
    setData(await res.json())
  }

  return (
    <div>
      <form onSubmit={handleSubmit}>
        <input type="file" accept="video/*" onChange={(e) => setFile(e.target.files[0])} />
        <button type="submit" disabled={!file}>Upload</button>
      </form>
      {data && <video src={data.video_url} controls />}
    </div>
  )
}

export default App
