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
  const path = window.location.pathname
  const page = path === '/predict'
    ? <PredictPage />
    : path === '/results'
      ? <ResultsPage />
      : path === '/models'
        ? <ModelsPage />
        : <Dashboard />

  return (
    <QueryClientProvider client={queryClient}>
      <div className="app">
        <nav className="navbar">
          <div className="nav-brand">
            <h1>Wafer Pattern Control Room</h1>
          </div>
          <div className="nav-links">
            <a href="/">Dashboard</a>
            <a href="/predict">Classify</a>
            <a href="/results">Results</a>
            <a href="/models">Model</a>
          </div>
        </nav>

        <main className="main-content">{page}</main>
      </div>
    </QueryClientProvider>
  )
}

export default App
