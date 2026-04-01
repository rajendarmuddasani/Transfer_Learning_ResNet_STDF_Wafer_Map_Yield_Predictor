import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import Dashboard from './pages/Dashboard'
import PredictPage from './pages/PredictPage'
import ResultsPage from './pages/ResultsPage'
import ModelsPage from './pages/ModelsPage'
import './App.css'

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: 1,
    },
  },
})

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <Router>
        <div className="app">
          <nav className="navbar">
            <div className="nav-brand">
              <h1>STDF Wafer Map Yield Predictor</h1>
            </div>
            <div className="nav-links">
              <a href="/">Dashboard</a>
              <a href="/predict">Predict</a>
              <a href="/results">Results</a>
              <a href="/models">Models</a>
            </div>
          </nav>
          
          <main className="main-content">
            <Routes>
              <Route path="/" element={<Dashboard />} />
              <Route path="/predict" element={<PredictPage />} />
              <Route path="/results" element={<ResultsPage />} />
              <Route path="/models" element={<ModelsPage />} />
            </Routes>
          </main>
        </div>
      </Router>
    </QueryClientProvider>
  )
}

export default App
