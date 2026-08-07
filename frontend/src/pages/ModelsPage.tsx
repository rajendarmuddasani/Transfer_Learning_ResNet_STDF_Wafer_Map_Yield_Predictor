import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'

export default function ModelsPage() {
  const { data: model, isLoading } = useQuery({
    queryKey: ['models'],
    queryFn: async () => (await api.listModels()).data[0],
  })

  return (
    <div>
      <div className="page-header">
        <p className="eyebrow">Read-only registry</p>
        <h2>Confirmed Model</h2>
        <p>Artifact identity and promotion evidence. Automated model promotion is not implemented.</p>
      </div>

      {isLoading ? <div className="loading"><div className="spinner" /></div> : model && (
        <section className="card model-record">
          <div className="model-record-header">
            <div>
              <span className="status-mark ready">{model.stage}</span>
              <h3>{model.model_name}</h3>
            </div>
            <span className="mono">{model.version}</span>
          </div>
          <dl className="fact-list">
            <div><dt>Architecture</dt><dd>{model.architecture}</dd></div>
            <div><dt>Accuracy</dt><dd>{model.accuracy ? `${(model.accuracy * 100).toFixed(2)}%` : 'Unavailable'}</dd></div>
            <div><dt>Macro F1</dt><dd>{model.macro_f1 ? model.macro_f1.toFixed(4) : 'Unavailable'}</dd></div>
            <div><dt>Minimum recall</dt><dd>{model.minimum_class_recall ? `${(model.minimum_class_recall * 100).toFixed(2)}%` : 'Unavailable'}</dd></div>
            <div><dt>SHA-256</dt><dd className="mono hash">{model.model_sha256}</dd></div>
            <div><dt>Data scope</dt><dd>{model.data_scope}</dd></div>
          </dl>
        </section>
      )}
    </div>
  )
}
