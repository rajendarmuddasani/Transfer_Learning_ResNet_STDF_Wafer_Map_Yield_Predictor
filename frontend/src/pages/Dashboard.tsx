import { useQuery } from '@tanstack/react-query'
import { api } from '../api/client'

export default function Dashboard() {
  const { data: healthData } = useQuery({
    queryKey: ['health'],
    queryFn: () => api.health(),
  })

  return (
    <div>
      <div className="page-header">
        <h2>Dashboard</h2>
        <p>Overview of wafer yield prediction system</p>
      </div>

      <div className="grid grid-3">
        <div className="card stat-card">
          <div className="stat-value">92.4%</div>
          <div className="stat-label">Model Accuracy</div>
        </div>
        <div className="card stat-card">
          <div className="stat-value">1,234</div>
          <div className="stat-label">Predictions Today</div>
        </div>
        <div className="card stat-card">
          <div className="stat-value">187ms</div>
          <div className="stat-label">Avg Inference Time</div>
        </div>
      </div>

      <div className="grid grid-2">
        <div className="card">
          <div className="card-header">Recent Predictions</div>
          <p>No recent predictions available.</p>
        </div>

        <div className="card">
          <div className="card-header">System Status</div>
          <p>Status: {healthData?.data?.status || 'Loading...'}</p>
          <p>Model Loaded: {healthData?.data?.model_loaded ? 'Yes' : 'No'}</p>
        </div>
      </div>
    </div>
  )
}
