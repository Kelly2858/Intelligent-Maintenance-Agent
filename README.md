# Task 1 — Intelligent Maintenance Agent

An AI-powered agent that processes natural language maintenance complaints for industrial equipment, classifies them, assigns priority, and stores results in a SQL database.

## Features

- **Natural Language Input** — Accepts free-text maintenance complaints
- **Issue Classification** — Categorizes into: `Electrical`, `Mechanical`, `Sensor`, `Unknown`
- **Priority Assignment** — Assigns: `Low`, `Medium`, `High` based on urgency signals
- **Structured JSON Output** — Returns detailed, machine-readable results
- **SQL Storage** — Persists all tickets in a SQLite database
- **Web Dashboard** — Beautiful dark-mode UI with real-time stats
- **REST API** — Full CRUD endpoints with filtering and pagination

## Architecture

```
Complaint (Text)
      │
      ▼
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────┐
│  Classifier  │───▶│  Prioritizer  │───▶│  SQL Store   │───▶│  JSON    │
│  (Keywords)  │    │  (Urgency)    │    │  (SQLite)    │    │  Output  │
└─────────────┘    └──────────────┘    └─────────────┘    └──────────┘
```

## Setup Instructions

### Prerequisites
- Python 3.10+

### Installation

```bash
# 1. Navigate to the task1 directory
cd task1

# 2. Create a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
uvicorn main:app --reload --port 8000
```

### Access
- **Web UI**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs (Swagger UI)
- **ReDoc**: http://localhost:8000/redoc

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/complaint` | Submit a maintenance complaint |
| `GET` | `/api/tickets` | List all tickets (with filters) |
| `GET` | `/api/tickets/{id}` | Get a specific ticket |
| `GET` | `/api/stats` | Dashboard statistics |

### Example API Call

```bash
curl -X POST http://localhost:8000/api/complaint \
  -H "Content-Type: application/json" \
  -d '{"complaint": "The circuit breaker tripped and there is a burning smell from the electrical panel"}'
```

### Example Response

```json
{
  "ticket_id": 1,
  "complaint": "The circuit breaker tripped and there is a burning smell from the electrical panel",
  "category": "Electrical",
  "priority": "High",
  "confidence": 0.85,
  "keywords_matched": ["circuit breaker", "electrical panel"],
  "priority_reasoning": "Urgent keywords detected: burning, tripped. Category 'Electrical' bias factor: 1.5x.",
  "classification_scores": {
    "Electrical": 0.85,
    "Mechanical": 0.1,
    "Sensor": 0.05
  }
}
```

## Project Structure

```
task1/
├── main.py                      # FastAPI application & routes
├── requirements.txt             # Python dependencies
├── README.md                    # This file
├── DOCUMENTATION.md             # Assumptions, trade-offs, improvements
├── sample_inputs_outputs.json   # Sample test cases
├── agent/
│   ├── __init__.py
│   ├── classifier.py            # Issue classification engine
│   ├── prioritizer.py           # Priority assignment engine
│   └── maintenance_agent.py     # Agent orchestrator
├── database/
│   ├── __init__.py
│   ├── db.py                    # SQLAlchemy connection setup
│   └── models.py                # ORM models
└── static/
    ├── index.html               # Web UI
    ├── style.css                # Styles
    └── app.js                   # Frontend logic
```

## Classification Approach

The agent uses a **weighted keyword-matching algorithm**:
1. Each category has a curated dictionary of domain-specific keywords with weights
2. Multi-word phrases are matched first (higher priority than single words)
3. Scores are normalized to produce a confidence value (0.0 – 1.0)
4. Category bias modifiers adjust priority (e.g., Electrical gets a 1.5x safety multiplier)

See [DOCUMENTATION.md](DOCUMENTATION.md) for detailed assumptions and production improvements.
