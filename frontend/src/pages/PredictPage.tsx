import { useState } from 'react'
import { useMutation } from '@tanstack/react-query'
import { api, PredictionResponse } from '../api/client'

export default function PredictPage() {
  const [file, setFile] = useState<File | null>(null)
  const [dragOver, setDragOver] = useState(false)
  const [result, setResult] = useState<PredictionResponse | null>(null)

  const predictMutation = useMutation({
    mutationFn: (formData: FormData) => api.predictSingle(formData),
    onSuccess: (response) => {
      setResult(response.data)
    },
  })

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0])
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setDragOver(false)
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0])
    }
  }

  const handleSubmit = async () => {
    if (!file) return

    const formData = new FormData()
    formData.append('stdf_file', file)
    formData.append('product_id', 'TC42x')
    formData.append('test_completion_pct', '10.0')

    predictMutation.mutate(formData)
  }

  return (
    <div>
      <div className="page-header">
        <h2>Predict Wafer Yield</h2>
        <p>Upload STDF file or wafer map image for prediction</p>
      </div>

      <div className="card">
        <div
          className={`upload-zone ${dragOver ? 'drag-over' : ''}`}
          onDragOver={(e) => {
            e.preventDefault()
            setDragOver(true)
          }}
          onDragLeave={() => setDragOver(false)}
          onDrop={handleDrop}
          onClick={() => document.getElementById('file-input')?.click()}
        >
          <input
            id="file-input"
            type="file"
            accept=".stdf,.std,.png,.jpg"
            onChange={handleFileChange}
            style={{ display: 'none' }}
          />
          {file ? (
            <div>
              <p>Selected: {file.name}</p>
              <p style={{ fontSize: '0.9rem', color: '#666' }}>
                Click to change or drag another file
              </p>
            </div>
          ) : (
            <div>
              <p style={{ fontSize: '1.2rem', marginBottom: '0.5rem' }}>
                Drop STDF file here or click to browse
              </p>
              <p style={{ fontSize: '0.9rem', color: '#666' }}>
                Supports .stdf, .std, .png, .jpg files
              </p>
            </div>
          )}
        </div>

        {file && (
          <div style={{ marginTop: '1rem', textAlign: 'center' }}>
            <button
              className="btn btn-primary"
              onClick={handleSubmit}
              disabled={predictMutation.isPending}
            >
              {predictMutation.isPending ? 'Predicting...' : 'Predict Yield'}
            </button>
          </div>
        )}
      </div>

      {predictMutation.isPending && (
        <div className="loading">
          <div className="spinner"></div>
        </div>
      )}

      {result && (
        <div className="card">
          <div className="card-header">Prediction Results</div>
          <div className="grid grid-2">
            <div>
              <p><strong>Wafer ID:</strong> {result.wafer_id}</p>
              <p><strong>Predicted Yield:</strong> {result.prediction.yield.toFixed(2)}%</p>
              <p><strong>Defect Class:</strong> {result.prediction.defect_class}</p>
              <p><strong>Confidence:</strong> {(result.prediction.confidence * 100).toFixed(2)}%</p>
            </div>
            <div>
              <p><strong>Model Version:</strong> {result.model_version}</p>
              <p><strong>Inference Time:</strong> {result.inference_time_ms.toFixed(0)}ms</p>
              <p><strong>Timestamp:</strong> {new Date(result.timestamp).toLocaleString()}</p>
            </div>
          </div>
        </div>
      )}

      {predictMutation.isError && (
        <div className="card" style={{ backgroundColor: '#ffebee', color: '#c62828' }}>
          <p><strong>Error:</strong> {(predictMutation.error as any)?.message || 'Prediction failed'}</p>
        </div>
      )}
    </div>
  )
}
