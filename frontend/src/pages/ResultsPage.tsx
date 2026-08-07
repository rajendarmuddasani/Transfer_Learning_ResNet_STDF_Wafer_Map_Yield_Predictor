export default function ResultsPage() {
  return (
    <div>
      <div className="page-header">
        <h2>Session Results</h2>
        <p>Persistent prediction history is not implemented.</p>
      </div>

      <div className="card">
        <div className="card-header">No stored records</div>
        <p>Classifications are returned to the caller and are not retained by this reference service.</p>
      </div>
    </div>
  )
}
