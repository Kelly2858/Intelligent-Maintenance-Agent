"""
SQLAlchemy ORM models for the Maintenance Agent database.
"""

from sqlalchemy import Column, Integer, String, Text, DateTime, Float
from sqlalchemy.sql import func
from database.db import Base


class MaintenanceTicket(Base):
    """
    Represents a processed maintenance complaint stored in the database.
    
    Each ticket captures:
    - The original natural language complaint
    - AI-classified category (Electrical / Mechanical / Sensor / Unknown)
    - AI-assigned priority (Low / Medium / High)
    - Confidence score of the classification
    - Extracted keywords that influenced the decision
    - Timestamp of creation
    """
    __tablename__ = "maintenance_tickets"

    id = Column(Integer, primary_key=True, autoincrement=True, index=True)
    complaint = Column(Text, nullable=False, comment="Original natural language complaint")
    category = Column(
        String(20),
        nullable=False,
        comment="Classified category: Electrical, Mechanical, Sensor, Unknown"
    )
    priority = Column(
        String(10),
        nullable=False,
        comment="Assigned priority: Low, Medium, High"
    )
    confidence = Column(
        Float,
        nullable=False,
        default=0.0,
        comment="Classification confidence score (0.0 to 1.0)"
    )
    keywords_matched = Column(
        Text,
        nullable=True,
        comment="Comma-separated keywords that triggered the classification"
    )
    created_at = Column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
        comment="Timestamp when the ticket was created"
    )

    def to_dict(self):
        """Serialize the ticket to a dictionary for JSON responses."""
        return {
            "id": self.id,
            "complaint": self.complaint,
            "category": self.category,
            "priority": self.priority,
            "confidence": round(self.confidence, 2),
            "keywords_matched": self.keywords_matched.split(", ") if self.keywords_matched else [],
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
