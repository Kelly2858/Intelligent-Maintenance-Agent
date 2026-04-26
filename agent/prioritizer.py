"""
Priority Assignment Module
===========================

Assigns a priority level (Low, Medium, High) to a maintenance complaint
based on keyword analysis and contextual urgency signals.

Priority Heuristics:
    HIGH   — Safety-critical, production-stopping, or emergency situations
    MEDIUM — Degraded performance, recurring issues, or gradual failures
    LOW    — Minor, cosmetic, or routine maintenance items

The module also considers the classified category as a contextual signal:
    - Electrical faults default higher (safety risk)
    - Unknown classifications default to Medium (err on the side of caution)
"""

import re
from dataclasses import dataclass


@dataclass
class PriorityResult:
    """Holds the result of priority assignment."""
    priority: str
    reasoning: str
    urgency_keywords: list[str]


# ---------------------------------------------------------------------------
# Urgency keyword dictionaries
# ---------------------------------------------------------------------------

HIGH_PRIORITY_KEYWORDS: dict[str, float] = {
    # Safety-critical
    "fire": 5.0,
    "smoke": 5.0,
    "explosion": 5.0,
    "electrocution": 5.0,
    "electric shock": 5.0,
    "safety hazard": 5.0,
    "safety": 3.0,
    "hazard": 4.0,
    "hazardous": 4.0,
    "danger": 4.0,
    "dangerous": 4.0,
    "toxic": 4.5,
    "gas leak": 5.0,
    "chemical leak": 5.0,
    "injury": 4.5,
    "burn": 3.5,
    "burning": 4.0,
    "burning smell": 4.5,
    # Production-stopping
    "shutdown": 4.0,
    "shut down": 4.0,
    "stopped": 3.0,
    "not working": 3.0,
    "completely failed": 4.5,
    "total failure": 4.5,
    "critical": 4.0,
    "emergency": 5.0,
    "urgent": 4.0,
    "immediately": 3.5,
    "catastrophic": 5.0,
    "production stopped": 5.0,
    "production down": 5.0,
    "line down": 4.5,
    "system down": 4.5,
    "outage": 3.5,
    "blackout": 4.0,
    "unresponsive": 3.0,
    "failed": 3.0,
    "failure": 3.0,
    "sparking": 4.0,
    "arcing": 4.0,
    "overheating": 3.5,
    "severe": 3.5,
}

MEDIUM_PRIORITY_KEYWORDS: dict[str, float] = {
    "intermittent": 3.0,
    "recurring": 3.0,
    "degraded": 3.0,
    "degradation": 3.0,
    "reduced": 2.5,
    "reduced performance": 3.5,
    "slow": 2.0,
    "slower": 2.0,
    "unusual": 2.5,
    "abnormal": 3.0,
    "erratic": 3.0,
    "fluctuating": 2.5,
    "inconsistent": 2.5,
    "unstable": 3.0,
    "gradually": 2.5,
    "getting worse": 3.5,
    "worsening": 3.5,
    "increasing": 2.0,
    "occasional": 2.5,
    "sometimes": 2.0,
    "partial": 2.5,
    "partially": 2.5,
    "noisy": 2.0,
    "loud": 2.0,
    "louder": 2.5,
    "vibrating": 2.5,
    "wear": 2.5,
    "worn": 2.5,
    "leaking": 2.5,
    "dripping": 2.5,
    "needs attention": 3.0,
    "requires maintenance": 3.0,
    "warning": 3.0,
    "warning light": 3.5,
    "alarm": 3.0,
    "error code": 3.0,
    "fault code": 3.0,
    "drift": 2.5,
    "misalignment": 2.5,
}

LOW_PRIORITY_KEYWORDS: dict[str, float] = {
    "minor": 3.0,
    "cosmetic": 3.5,
    "routine": 3.0,
    "scheduled": 3.0,
    "preventive": 3.0,
    "preventative": 3.0,
    "planned": 2.5,
    "regular": 2.0,
    "check": 1.5,
    "inspection": 2.5,
    "small": 2.0,
    "slight": 2.5,
    "barely": 2.5,
    "minimal": 2.5,
    "negligible": 3.0,
    "paint": 3.0,
    "scratch": 2.5,
    "label": 2.5,
    "clean": 2.0,
    "cleaning": 2.5,
    "tighten": 2.0,
    "adjustment": 2.0,
    "lubrication": 2.0,
    "top up": 2.0,
    "refill": 2.0,
    "replace filter": 2.5,
    "filter": 1.5,
    "when convenient": 3.5,
    "not urgent": 4.0,
    "low priority": 4.0,
    "non critical": 3.5,
}

# Category-based priority modifiers
CATEGORY_PRIORITY_BIAS = {
    "Electrical": 1.5,   # Electrical issues have higher inherent risk
    "Mechanical": 1.0,   # Neutral
    "Sensor": 0.8,       # Sensor issues are often less immediately dangerous
    "Unknown": 1.2,      # Unknown = err on the side of caution
}


def _preprocess_text(text: str) -> str:
    """Normalize input text for matching."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _score_priority(text: str, keywords: dict[str, float]) -> tuple[float, list[str]]:
    """Score text against a priority keyword dictionary."""
    total = 0.0
    matched = []
    sorted_kw = sorted(keywords.keys(), key=len, reverse=True)

    matched_positions: set[tuple[int, int]] = set()

    for kw in sorted_kw:
        pattern = r"\b" + re.escape(kw) + r"\b"
        for match in re.finditer(pattern, text):
            start, end = match.start(), match.end()
            overlap = any(start < me and end > ms for ms, me in matched_positions)
            if not overlap:
                matched_positions.add((start, end))
                total += keywords[kw]
                if kw not in matched:
                    matched.append(kw)

    return total, matched


def assign_priority(complaint: str, category: str) -> PriorityResult:
    """
    Assign a priority level to a maintenance complaint.
    
    Algorithm:
        1. Score the complaint against High, Medium, and Low keyword sets
        2. Apply a category-based bias multiplier (e.g., Electrical gets boosted)
        3. Select the priority level with the highest weighted score
        4. Default to Medium if no strong signals are found
    
    Args:
        complaint: Raw natural language complaint text
        category:  The classified category (used for bias adjustment)
        
    Returns:
        PriorityResult with priority level, reasoning, and matched keywords
    """
    processed = _preprocess_text(complaint)

    high_score, high_matched = _score_priority(processed, HIGH_PRIORITY_KEYWORDS)
    med_score, med_matched = _score_priority(processed, MEDIUM_PRIORITY_KEYWORDS)
    low_score, low_matched = _score_priority(processed, LOW_PRIORITY_KEYWORDS)

    # Apply category bias to the high-priority score
    cat_bias = CATEGORY_PRIORITY_BIAS.get(category, 1.0)
    high_score *= cat_bias

    # Decision logic
    all_keywords = high_matched + med_matched + low_matched

    # If we have any high-priority signals, it's High
    if high_score > 0 and high_score >= med_score and high_score >= low_score:
        return PriorityResult(
            priority="High",
            reasoning=f"Urgent keywords detected: {', '.join(high_matched)}. "
                      f"Category '{category}' bias factor: {cat_bias}x.",
            urgency_keywords=high_matched,
        )

    # If medium signals dominate
    if med_score > 0 and med_score >= low_score:
        return PriorityResult(
            priority="Medium",
            reasoning=f"Degraded performance indicators detected: {', '.join(med_matched)}.",
            urgency_keywords=med_matched,
        )

    # If low signals dominate
    if low_score > 0:
        return PriorityResult(
            priority="Low",
            reasoning=f"Routine/minor maintenance indicators: {', '.join(low_matched)}.",
            urgency_keywords=low_matched,
        )

    # Default: no urgency keywords found → use category-based default
    category_defaults = {
        "Electrical": "Medium",
        "Mechanical": "Medium",
        "Sensor": "Low",
        "Unknown": "Medium",
    }
    default_priority = category_defaults.get(category, "Medium")

    return PriorityResult(
        priority=default_priority,
        reasoning=f"No explicit urgency keywords detected. "
                  f"Defaulting to '{default_priority}' based on category '{category}'.",
        urgency_keywords=[],
    )
