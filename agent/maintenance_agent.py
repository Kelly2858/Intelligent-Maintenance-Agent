"""
Maintenance Agent Orchestrator
===============================

The central "agent" that orchestrates the full processing pipeline:

    1. RECEIVE  → Accept a natural language maintenance complaint
    2. CLASSIFY → Determine issue category (Electrical / Mechanical / Sensor / Unknown)
    3. PRIORITIZE → Assign urgency level (Low / Medium / High)
    4. STORE    → Persist the structured result in the SQL database
    5. RESPOND  → Return a structured JSON output

This follows an agentic pattern where each step is a distinct,
composable "tool" that the agent calls in sequence.
"""

from dataclasses import dataclass, asdict
from sqlalchemy.orm import Session
import os
import joblib

from agent.classifier import classify_complaint, ClassificationResult
from agent.prioritizer import assign_priority, PriorityResult
from database.models import MaintenanceTicket


@dataclass
class AgentResponse:
    """The structured output returned by the maintenance agent."""
    ticket_id: int
    complaint: str
    category: str
    priority: str
    confidence: float
    keywords_matched: list[str]
    priority_reasoning: str
    classification_scores: dict[str, float]

    def to_dict(self) -> dict:
        """Return a professionally nested structured JSON output."""
        return {
            "ticket_id": self.ticket_id,
            "status": "processed",
            "input": {
                "complaint": self.complaint,
            },
            "classification": {
                "category": self.category,
                "confidence": self.confidence,
                "confidence_pct": f"{round(self.confidence * 100)}%",
                "scores": self.classification_scores,
            },
            "priority_assessment": {
                "priority": self.priority,
                "reasoning": self.priority_reasoning,
            },
            "analysis": {
                "keywords_matched": self.keywords_matched,
                "keyword_count": len(self.keywords_matched),
            },
        }


class MaintenanceAgent:
    """
    The Intelligent Maintenance Agent.
    
    Processes natural language maintenance complaints through a
    multi-step pipeline: classify → prioritize → store → respond.
    
    Usage:
        agent = MaintenanceAgent(db_session)
        result = agent.process_complaint("The motor is overheating and sparking")
    """

    def __init__(self, db: Session):
        self.db = db
        
        # Load ML models if they exist
        model_dir = os.path.join(os.path.dirname(__file__), "models")
        cat_path = os.path.join(model_dir, "category_model.pkl")
        pri_path = os.path.join(model_dir, "priority_model.pkl")
        
        self.ml_category = joblib.load(cat_path) if os.path.exists(cat_path) else None
        self.ml_priority = joblib.load(pri_path) if os.path.exists(pri_path) else None

    def process_complaint(self, complaint: str) -> AgentResponse:
        """
        Process a single maintenance complaint through the full agent pipeline.
        
        Args:
            complaint: Natural language description of the maintenance issue
            
        Returns:
            AgentResponse with all structured fields
            
        Raises:
            ValueError: If the complaint is empty or whitespace-only
        """
        # --- Step 0: Validate input ---
        complaint = complaint.strip()
        if not complaint:
            raise ValueError("Complaint text cannot be empty.")

        # --- Step 1: CLASSIFY the issue ---
        if self.ml_category:
            predicted_cat = str(self.ml_category.predict([complaint])[0])
            probs = self.ml_category.predict_proba([complaint])[0]
            classes = self.ml_category.classes_
            
            # Map probabilities to all known classes for the UI score bars
            all_scores = {str(c): float(p) for c, p in zip(classes, probs)}
            cat_prob = float(max(probs))
            
            classification = ClassificationResult(
                category=predicted_cat,
                confidence=cat_prob,
                keywords_matched=["ML Model Prediction"],
                all_scores=all_scores
            )
        else:
            classification = classify_complaint(complaint)

        # --- Step 2: PRIORITIZE the issue ---
        if self.ml_priority:
            predicted_pri = str(self.ml_priority.predict([complaint])[0])
            priority = PriorityResult(
                priority=predicted_pri,
                reasoning="Assigned by trained Machine Learning model based on historical data.",
                urgency_keywords=["ML Model Prediction"]
            )
        else:
            priority = assign_priority(complaint, classification.category)

        # --- Step 3: STORE in the SQL database ---
        ticket = MaintenanceTicket(
            complaint=complaint,
            category=classification.category,
            priority=priority.priority,
            confidence=classification.confidence,
            keywords_matched=", ".join(classification.keywords_matched),
        )
        self.db.add(ticket)
        self.db.commit()
        self.db.refresh(ticket)

        # --- Step 4: BUILD structured response ---
        response = AgentResponse(
            ticket_id=ticket.id,
            complaint=complaint,
            category=classification.category,
            priority=priority.priority,
            confidence=round(classification.confidence, 2),
            keywords_matched=classification.keywords_matched,
            priority_reasoning=priority.reasoning,
            classification_scores={
                k: round(v, 3) for k, v in classification.all_scores.items()
            },
        )

        return response
