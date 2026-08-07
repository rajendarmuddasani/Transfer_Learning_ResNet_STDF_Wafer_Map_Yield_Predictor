import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { api, PredictionResponse } from '../api/client'

export default function PredictPage() {
  const [file, setFile] = useState<File | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const [result, setResult] = useState<PredictionResponse | null>(null)

  const predictMutation = useMutation({
    mutationFn: (formData: FormData) => api.predictSingle(formData),
    onSuccess: (response) => setResult(response.data),
  })

  const chooseFile = (nextFile?: File) => {
    if (nextFile && ['image/png', 'image/jpeg'].includes(nextFile.type)) {
      setFile(nextFile)
      setResult(null)
    }
  }

  const handleSubmit = () => {
    if (!file) return
    const formData = new FormData()
    formData.append('wafer_map_image', file)
    predictMutation.mutate(formData)
  }

  const rankedProbabilities = result
    ? Object.entries(result.defect_probabilities).sort(([, left], [, right]) => right - left)
    : []

  return (
    <div>
      <div className="page-header">
        <p className="eyebrow">Single bounded decision</p>
        <h2>Classify Wafer Map</h2>
        <p>Upload one PNG or JPEG wafer-map image. STDF ingestion and batch jobs are outside this confirmed runtime.</p>
      </div>

      <div className="tool-layout">
        <section className="card">
          <div
            className={`upload-zone ${dragOver ? 'drag-over' : ''}`}
            onDragOver={(event) => { event.preventDefault(); setDragOver(true) }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(event) => { event.preventDefault(); setDragOver(false); chooseFile(event.dataTransfer.files[0]) }}
            onClick={() => document.getElementById('file-input')?.click()}
          >
            <input
              id="file-input"
              type="file"
              accept="image/png,image/jpeg"
              onChange={(event) => chooseFile(event.target.files?.[0])}
              style={{ display: 'none' }}
            />
            <span className="upload-icon">+</span>
            <strong>{file ? file.name : 'Select wafer-map image'}</strong>
            <small>PNG or JPEG, maximum 10 MiB</small>
          </div>
          <button className="btn btn-primary" onClick={handleSubmit} disabled={!file || predictMutation.isPending}>
            {predictMutation.isPending ? 'Classifying' : 'Run classifier'}
          </button>
          {predictMutation.isError && <p className="error-text">Classification failed. Verify the API is ready and the image is valid.</p>}
        </section>

        <section className="card result-panel">
          {result ? (
            <>
              <p className="eyebrow">Confirmed model response</p>
              <h3>{result.defect_class}</h3>
              <p className="confidence">{(result.confidence * 100).toFixed(2)}% confidence</p>
              <div className="probability-list">
                {rankedProbabilities.map(([className, probability]) => (
                  <div key={className}>
                    <span>{className}</span>
                    <div className="probability-track"><i style={{ width: `${probability * 100}%` }} /></div>
                    <strong>{(probability * 100).toFixed(1)}%</strong>
                  </div>
                ))}
              </div>
              <dl className="fact-list compact">
                <div><dt>Model</dt><dd>{result.model_version}</dd></div>
                <div><dt>Latency</dt><dd>{result.inference_time_ms.toFixed(2)} ms</dd></div>
                <div><dt>Request</dt><dd className="mono">{result.request_id.slice(0, 14)}...</dd></div>
              </dl>
            </>
          ) : (
            <div className="empty-result">
              <strong>No classification yet</strong>
              <p>The result will include all eight calibrated class probabilities and the exact model identity.</p>
            </div>
          )}
        </section>
      </div>
    </div>
  )
}
