import { useQuery } from '@tanstack/react-query'
import { api, ModelInfo } from '../api/client'

export default function ModelsPage() {
  const { data: models, isLoading } = useQuery({
    queryKey: ['models'],
    queryFn: async () => {
      const response = await api.listModels()
      return response.data
    },
  })

  return (
    <div>
      <div className="page-header">
        <h2>Model Management</h2>
        <p>View and manage deployed models</p>
      </div>

      {isLoading && (
        <div className="loading">
          <div className="spinner"></div>
        </div>
      )}

      {models && (
        <div className="grid grid-2">
          {models.map((model: ModelInfo) => (
            <div key={model.model_id} className="card">
              <div className="card-header">{model.model_name}</div>
              <p><strong>Architecture:</strong> {model.architecture}</p>
              <p><strong>Version:</strong> {model.version}</p>
              <p><strong>Stage:</strong> <span style={{
                padding: '0.25rem 0.5rem',
                borderRadius: '4px',
                backgroundColor: model.stage === 'PRODUCTION' ? '#4caf50' : '#ff9800',
                color: 'white',
                fontSize: '0.85rem'
              }}>{model.stage}</span></p>
              {model.accuracy && (
                <p><strong>Accuracy:</strong> {(model.accuracy * 100).toFixed(2)}%</p>
              )}
              <p><strong>Created:</strong> {new Date(model.created_at).toLocaleDateString()}</p>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
