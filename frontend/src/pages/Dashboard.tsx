import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'

export default function Dashboard() {
  const { data: healthData } = useQuery({
    queryKey: ['health'],
    queryFn: () => api.health(),
  })
  const { data: modelData } = useQuery({
    queryKey: ['models'],
    queryFn: async () => (await api.listModels()).data[0],
  })

  const metric = (value?: number) => value === undefined ? 'Loading' : `${(value * 100).toFixed(2)}%`

  return (
    <div>
      <div className="page-header evidence-header">
        <div>
          <p className="eyebrow">Confirmed synthetic evidence</p>
          <h2>Wafer Pattern Control Room</h2>
          <p>Hash-bound ResNet-18 classification with grouped confirmation and explicit operating limits.</p>
        </div>
        <div className={`status-mark ${healthData?.data?.model_loaded ? 'ready' : 'not-ready'}`}>
          {healthData?.data?.model_loaded ? 'Model ready' : 'Model unavailable'}
        </div>
      </div>

      <div className="metric-grid">
        <div className="metric-panel">
          <span>Confirmation accuracy</span>
          <strong>{metric(modelData?.accuracy)}</strong>
          <small>800 disjoint synthetic wafers</small>
        </div>
        <div className="metric-panel">
          <span>Macro F1</span>
          <strong>{metric(modelData?.macro_f1)}</strong>
          <small>Eight balanced classes</small>
        </div>
        <div className="metric-panel tail">
          <span>Minimum class recall</span>
          <strong>{metric(modelData?.minimum_class_recall)}</strong>
          <small>QuadrantFailure is the tail</small>
        </div>
      </div>

      <div className="grid grid-2">
        <section className="card">
          <div className="card-header">Runtime identity</div>
          <dl className="fact-list">
            <div><dt>Version</dt><dd>{modelData?.version || 'Loading'}</dd></div>
            <div><dt>Stage</dt><dd>{modelData?.stage || 'Loading'}</dd></div>
            <div><dt>SHA-256</dt><dd className="mono">{modelData?.model_sha256?.slice(0, 20) || 'Loading'}...</dd></div>
          </dl>
        </section>
        <section className="card boundary-card">
          <div className="card-header">Confirmed boundary</div>
          <p>Independent synthetic wafer maps only. This evidence does not establish WM-811K quality, production silicon performance, STDF parsing correctness, or yield prediction.</p>
        </section>
      </div>
    </div>
  )
}
