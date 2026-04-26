"""
Intelligent Maintenance Agent — FastAPI Application
=====================================================

REST API endpoints:
    POST /api/complaint      → Submit a maintenance complaint
    GET  /api/tickets         → Retrieve all stored tickets
    GET  /api/tickets/{id}    → Retrieve a specific ticket
    GET  /api/stats           → Get dashboard statistics
    GET  /                    → Serve the web UI

Run with:
    uvicorn main:app --reload --port 8000
"""

from fastapi import FastAPI, Depends, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel, Field
import os

from database.db import get_db, init_db
from database.models import MaintenanceTicket
from agent.maintenance_agent import MaintenanceAgent
from agent.ml_trainer import train_agent_models

# ---------------------------------------------------------------------------
# Application setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Intelligent Maintenance Agent",
    description="AI-powered maintenance complaint classifier and priority assigner",
    version="1.0.0",
)

# Mount static files
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
def on_startup():
    """Initialize the database tables on application startup."""
    init_db()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------

class ComplaintRequest(BaseModel):
    """Request body for submitting a maintenance complaint."""
    complaint: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        description="Natural language description of the maintenance issue",
        examples=["The conveyor belt motor is making a loud grinding noise and overheating"],
    )


class ClassificationSchema(BaseModel):
    """Classification sub-schema."""
    category: str
    confidence: float
    confidence_pct: str
    scores: dict[str, float]


class PrioritySchema(BaseModel):
    """Priority assessment sub-schema."""
    priority: str
    reasoning: str


class AnalysisSchema(BaseModel):
    """Analysis sub-schema."""
    keywords_matched: list[str]
    keyword_count: int


class InputSchema(BaseModel):
    """Input sub-schema."""
    complaint: str


class TicketResponse(BaseModel):
    """Response schema for a processed maintenance ticket — structured JSON output."""
    ticket_id: int
    status: str
    input: InputSchema
    classification: ClassificationSchema
    priority_assessment: PrioritySchema
    analysis: AnalysisSchema


class StatsResponse(BaseModel):
    """Dashboard statistics."""
    total_tickets: int
    by_category: dict[str, int]
    by_priority: dict[str, int]


class BulkDeleteRequest(BaseModel):
    """Request schema for deleting multiple tickets."""
    ticket_ids: list[int]


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------

@app.post("/api/complaint", response_model=TicketResponse, status_code=201)
def submit_complaint(request: ComplaintRequest, db: Session = Depends(get_db)):
    """
    Submit a natural language maintenance complaint.
    
    The AI agent will:
    1. Classify the issue (Electrical / Mechanical / Sensor / Unknown)
    2. Assign a priority (Low / Medium / High)
    3. Store the result in the database
    4. Return structured JSON output
    """
    try:
        agent = MaintenanceAgent(db)
        result = agent.process_complaint(request.complaint)
        return result.to_dict()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/tickets", response_model=list[dict])
def get_all_tickets(
    skip: int = 0,
    limit: int = 50,
    category: str | None = None,
    priority: str | None = None,
    db: Session = Depends(get_db),
):
    """
    Retrieve all maintenance tickets with optional filtering.
    
    Query params:
        skip     — Offset for pagination (default 0)
        limit    — Max results to return (default 50)
        category — Filter by category (Electrical, Mechanical, Sensor, Unknown)
        priority — Filter by priority (Low, Medium, High)
    """
    query = db.query(MaintenanceTicket)

    if category:
        query = query.filter(MaintenanceTicket.category == category)
    if priority:
        query = query.filter(MaintenanceTicket.priority == priority)

    tickets = query.order_by(MaintenanceTicket.id.desc()).offset(skip).limit(limit).all()
    return [t.to_dict() for t in tickets]


@app.post("/api/tickets/bulk")
def delete_bulk_tickets(req: BulkDeleteRequest, db: Session = Depends(get_db)):
    """Delete specifically selected tickets from the history."""
    deleted_count = db.query(MaintenanceTicket).filter(MaintenanceTicket.id.in_(req.ticket_ids)).delete(synchronize_session=False)
    db.commit()
    
    # If the database has less than 3 tickets now, we should reset the ML models
    remaining = db.query(MaintenanceTicket).count()
    if remaining < 3:
        model_dir = os.path.join(os.path.dirname(__file__), "agent", "models")
        cat_path = os.path.join(model_dir, "category_model.pkl")
        pri_path = os.path.join(model_dir, "priority_model.pkl")
        if os.path.exists(cat_path): os.remove(cat_path)
        if os.path.exists(pri_path): os.remove(pri_path)
        
    return {"message": f"Deleted {deleted_count} tickets."}


@app.get("/api/tickets/{ticket_id}", response_model=dict)
def get_ticket(ticket_id: int, db: Session = Depends(get_db)):
    """Retrieve a specific maintenance ticket by ID."""
    ticket = db.query(MaintenanceTicket).filter(MaintenanceTicket.id == ticket_id).first()
    if not ticket:
        raise HTTPException(status_code=404, detail=f"Ticket #{ticket_id} not found")
    return ticket.to_dict()


@app.get("/api/stats", response_model=StatsResponse)
def get_stats(db: Session = Depends(get_db)):
    """Get dashboard statistics: total tickets, breakdown by category and priority."""
    total = db.query(MaintenanceTicket).count()

    # Group by category
    cat_rows = (
        db.query(MaintenanceTicket.category, func.count(MaintenanceTicket.id))
        .group_by(MaintenanceTicket.category)
        .all()
    )
    by_category = {row[0]: row[1] for row in cat_rows}

    # Group by priority
    pri_rows = (
        db.query(MaintenanceTicket.priority, func.count(MaintenanceTicket.id))
        .group_by(MaintenanceTicket.priority)
        .all()
    )
    by_priority = {row[0]: row[1] for row in pri_rows}

    return StatsResponse(
        total_tickets=total,
        by_category=by_category,
        by_priority=by_priority,
    )


@app.post("/api/train")
def train_agent(db: Session = Depends(get_db)):
    """Train the Machine Learning models based on the accumulated ticket history."""
    try:
        result = train_agent_models(db)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))






@app.get("/api/samples")
def get_samples():
    """Retrieve the sample inputs and outputs from the JSON file."""
    samples_path = os.path.join(os.path.dirname(__file__), "sample_inputs_outputs.json")
    if not os.path.exists(samples_path):
        return []
    import json
    with open(samples_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Web UI route
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
def serve_ui():
    """Serve the main web UI."""
    html_path = os.path.join(STATIC_DIR, "index.html")
    with open(html_path, "r", encoding="utf-8") as f:
        return HTMLResponse(content=f.read())
