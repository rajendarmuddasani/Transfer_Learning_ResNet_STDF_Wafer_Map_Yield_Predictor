/**
 * API Client for P02 Yield Predictor
 */

import axios from 'axios'

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8001'

const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
})

// Request interceptor for adding auth token
apiClient.interceptors.request.use(
  (config) => {
    const token = localStorage.getItem('token')
    if (token) {
      config.headers.Authorization = `Bearer ${token}`
    }
    return config
  },
  (error) => Promise.reject(error)
)

// Response interceptor for error handling
apiClient.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      // Handle unauthorized
      localStorage.removeItem('token')
      window.location.href = '/login'
    }
    return Promise.reject(error)
  }
)

export interface PredictionResponse {
  wafer_id: string
  prediction: {
    yield: number
    defect_class: string
    defect_probabilities: Record<string, number>
    confidence: number
    uncertainty: number
  }
  model_version: string
  inference_time_ms: number
  grad_cam_url?: string
  timestamp: string
}

export interface ModelInfo {
  model_id: string
  model_name: string
  architecture: string
  version: string
  stage: string
  accuracy?: number
  created_at: string
  created_by: string
}

export const api = {
  // Health check
  health: () => apiClient.get('/api/v1/health'),

  // Predictions
  predictSingle: (formData: FormData) =>
    apiClient.post<PredictionResponse>('/api/v1/predict', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    }),

  predictBatch: (data: { wafer_ids?: string[]; lot_id?: string }) =>
    apiClient.post('/api/v1/predict/batch', data),

  getJobStatus: (jobId: string) =>
    apiClient.get(`/api/v1/jobs/${jobId}`),

  getBatchResults: (jobId: string) =>
    apiClient.get(`/api/v1/results/${jobId}`),

  // Models
  listModels: () =>
    apiClient.get<ModelInfo[]>('/api/v1/models'),

  getModel: (modelId: string) =>
    apiClient.get<ModelInfo>(`/api/v1/models/${modelId}`),

  promoteModel: (modelId: string, data: any) =>
    apiClient.post(`/api/v1/models/${modelId}/promote`, data),
}

export default apiClient
