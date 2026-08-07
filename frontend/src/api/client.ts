/** Typed fetch client for the confirmed wafer classifier API. */

export interface PredictionResponse {
  wafer_reference: string
  task: string
  defect_class: string
  defect_probabilities: Record<string, number>
  confidence: number
  model_version: string
  model_sha256: string
  inference_time_ms: number
  request_id: string
  timestamp: string
}

export interface ModelInfo {
  model_id: string
  model_name: string
  architecture: string
  version: string
  stage: string
  accuracy?: number
  macro_f1?: number
  minimum_class_recall?: number
  model_sha256: string
  data_scope: string
}

interface ApiResponse<T> {
  data: T
}

async function request<T>(path: string, options?: RequestInit): Promise<ApiResponse<T>> {
  const headers = new Headers(options?.headers)
  const token = localStorage.getItem('token')
  if (token) headers.set('Authorization', `Bearer ${token}`)
  const response = await fetch(path, { ...options, headers })
  if (response.status === 401) localStorage.removeItem('token')
  if (!response.ok) {
    const detail = await response.json().catch(() => ({ detail: 'Request failed' }))
    throw new Error(detail.detail || `Request failed with status ${response.status}`)
  }
  return { data: await response.json() as T }
}

export const api = {
  health: () => request<{ status: string; model_loaded: boolean }>('/api/v1/health'),
  predictSingle: (formData: FormData) => request<PredictionResponse>('/api/v1/classify-image', {
    method: 'POST',
    body: formData,
  }),
  listModels: () => request<ModelInfo[]>('/api/v1/models'),
  getModel: (modelId: string) => request<ModelInfo>(`/api/v1/models/${encodeURIComponent(modelId)}`),
}
