"""
Issue Classifier Module
=======================

Classifies maintenance complaints into categories:
  - Electrical
  - Mechanical
  - Sensor
  - Unknown

Uses a weighted keyword-matching algorithm with contextual scoring.
Each keyword carries a base weight, and the classifier computes a
normalized confidence score per category. The category with the highest
aggregate score wins. If no category exceeds the minimum threshold,
the complaint is classified as "Unknown".

Design Decision:
    A rule-based approach was chosen over an LLM call for classification
    because it provides deterministic, explainable results with zero
    latency and no API cost. In production, this could be swapped for
    a fine-tuned transformer model or an LLM API call.
"""

import re
from dataclasses import dataclass


@dataclass
class ClassificationResult:
    """Holds the result of a complaint classification."""
    category: str
    confidence: float
    keywords_matched: list[str]
    all_scores: dict[str, float]


# ---------------------------------------------------------------------------
# Keyword dictionaries with weights
# Higher weight = stronger signal for that category
# ---------------------------------------------------------------------------

ELECTRICAL_KEYWORDS: dict[str, float] = {
    # Core electrical terms
    "electrical": 3.0,
    "voltage": 3.0,
    "current": 2.0,
    "circuit": 3.0,
    "wiring": 3.0,
    "wire": 2.5,
    "power supply": 3.5,
    "power": 2.0,
    "fuse": 3.0,
    "breaker": 3.0,
    "circuit breaker": 3.5,
    "transformer": 3.0,
    "capacitor": 3.0,
    "resistor": 2.5,
    "inverter": 3.0,
    "rectifier": 3.0,
    "relay": 2.5,
    "contactor": 3.0,
    "grounding": 3.0,
    "ground fault": 3.5,
    "short circuit": 3.5,
    "short": 1.5,
    "overload": 2.5,
    "overcurrent": 3.0,
    "arc": 2.5,
    "arcing": 3.0,
    "spark": 2.5,
    "sparking": 3.0,
    "insulation": 2.5,
    "conductor": 2.5,
    "phase": 2.0,
    "three phase": 3.0,
    "single phase": 3.0,
    "ampere": 2.5,
    "amp": 2.0,
    "watt": 2.0,
    "ohm": 2.5,
    "impedance": 3.0,
    "inductance": 3.0,
    "electrocution": 3.5,
    "shock": 2.5,
    "electric shock": 3.5,
    "tripped": 2.5,
    "tripping": 2.5,
    "blackout": 3.0,
    "power outage": 3.5,
    "power failure": 3.5,
    "battery": 2.0,
    "generator": 2.5,
    "motor winding": 3.0,
    "coil": 2.0,
    "solenoid": 2.5,
    "switchgear": 3.0,
    "panel": 0.8,
    "electrical panel": 3.5,
    "distribution board": 3.0,
    "vfd": 3.0,
    "variable frequency drive": 3.5,
    "plc": 2.0,
}

MECHANICAL_KEYWORDS: dict[str, float] = {
    # Core mechanical terms
    "mechanical": 3.0,
    "vibration": 3.0,
    "vibrating": 3.0,
    "bearing": 3.0,
    "gear": 3.0,
    "gearbox": 3.5,
    "alignment": 2.5,
    "misalignment": 3.0,
    "shaft": 3.0,
    "coupling": 2.5,
    "belt": 2.5,
    "belt drive": 3.0,
    "pulley": 3.0,
    "chain": 2.0,
    "sprocket": 3.0,
    "wear": 2.0,
    "worn": 2.5,
    "worn out": 3.0,
    "tear": 2.0,
    "crack": 2.5,
    "cracked": 2.5,
    "fracture": 3.0,
    "fatigue": 2.5,
    "metal fatigue": 3.5,
    "corrosion": 2.5,
    "rust": 2.5,
    "erosion": 2.5,
    "leak": 2.0,
    "leaking": 2.5,
    "oil leak": 3.0,
    "hydraulic": 3.0,
    "hydraulic fluid": 3.0,
    "pneumatic": 3.0,
    "piston": 3.0,
    "cylinder": 2.5,
    "valve": 2.5,
    "pump": 2.5,
    "compressor": 2.5,
    "turbine": 3.0,
    "fan blade": 3.0,
    "impeller": 3.0,
    "lubrication": 3.0,
    "lubricant": 3.0,
    "grease": 2.5,
    "friction": 2.5,
    "seizure": 3.0,
    "seized": 3.0,
    "jam": 2.0,
    "jammed": 2.5,
    "stuck": 2.0,
    "broken": 2.0,
    "bent": 2.5,
    "deformed": 2.5,
    "noise": 1.5,
    "grinding": 2.5,
    "grinding noise": 3.0,
    "squealing": 2.5,
    "rattling": 2.5,
    "clunking": 2.5,
    "knocking": 2.5,
    "torque": 2.5,
    "rpm": 2.0,
    "speed": 1.5,
    "overheating": 2.0,
    "motor": 1.5,
    "conveyor": 2.5,
    "conveyor belt": 3.0,
    "roller": 2.5,
    "spindle": 3.0,
}

SENSOR_KEYWORDS: dict[str, float] = {
    # Core sensor terms
    "sensor": 3.5,
    "sensors": 3.5,
    "calibration": 3.5,
    "calibrate": 3.0,
    "recalibrate": 3.5,
    "recalibration": 3.5,
    "reading": 2.0,
    "readings": 2.0,
    "false reading": 3.5,
    "false readings": 3.5,
    "inaccurate reading": 3.5,
    "erratic reading": 3.5,
    "wrong reading": 3.0,
    "incorrect reading": 3.0,
    "signal": 2.5,
    "signal loss": 3.5,
    "no signal": 3.5,
    "weak signal": 3.0,
    "drift": 3.0,
    "sensor drift": 3.5,
    "measurement": 2.5,
    "temperature sensor": 3.5,
    "pressure sensor": 3.5,
    "proximity sensor": 3.5,
    "level sensor": 3.5,
    "flow sensor": 3.5,
    "speed sensor": 3.5,
    "vibration sensor": 3.0,
    "thermocouple": 3.5,
    "rtd": 3.0,
    "transducer": 3.5,
    "transmitter": 2.5,
    "probe": 2.5,
    "gauge": 2.5,
    "pressure gauge": 3.0,
    "flowmeter": 3.5,
    "flow meter": 3.5,
    "encoder": 3.0,
    "accelerometer": 3.5,
    "gyroscope": 3.5,
    "detector": 2.5,
    "photoelectric": 3.5,
    "ultrasonic": 3.0,
    "infrared": 3.0,
    "laser sensor": 3.5,
    "feedback": 2.0,
    "anomalous": 2.5,
    "fluctuating": 2.5,
    "intermittent signal": 3.0,
    "data logger": 3.0,
    "monitoring": 1.5,
    "display": 1.5,
    "readout": 2.5,
    "zero point": 3.0,
    "offset": 2.5,
    "saturation": 2.5,
    "out of range": 3.0,
    "malfunction": 1.5,
}

# Mapping of category names to their keyword dictionaries
CATEGORY_KEYWORDS = {
    "Electrical": ELECTRICAL_KEYWORDS,
    "Mechanical": MECHANICAL_KEYWORDS,
    "Sensor": SENSOR_KEYWORDS,
}

# Minimum confidence threshold to avoid classifying noise
MIN_CONFIDENCE_THRESHOLD = 0.10


def _preprocess_text(text: str) -> str:
    """
    Normalize the input text for keyword matching.
    - Lowercase
    - Remove special characters (keep alphanumeric and spaces)
    - Collapse multiple spaces
    """
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def _score_category(text: str, keywords: dict[str, float]) -> tuple[float, list[str]]:
    """
    Score a preprocessed text against a keyword dictionary.
    
    Uses multi-word phrase matching first (higher priority),
    then falls back to single-word matching.
    
    Returns:
        tuple of (total_score, list_of_matched_keywords)
    """
    total_score = 0.0
    matched = []

    # Sort keywords by length (longest first) so multi-word phrases match before substrings
    sorted_keywords = sorted(keywords.keys(), key=len, reverse=True)

    # Track which portions of text have been matched to avoid double-counting
    matched_positions: set[tuple[int, int]] = set()

    for keyword in sorted_keywords:
        # Find all occurrences of the keyword in the text
        pattern = r"\b" + re.escape(keyword) + r"\b"
        for match in re.finditer(pattern, text):
            start, end = match.start(), match.end()

            # Check if this position overlaps with an already matched phrase
            overlap = False
            for ms, me in matched_positions:
                if start < me and end > ms:
                    overlap = True
                    break

            if not overlap:
                matched_positions.add((start, end))
                total_score += keywords[keyword]
                if keyword not in matched:
                    matched.append(keyword)

    return total_score, matched


def classify_complaint(complaint: str) -> ClassificationResult:
    """
    Classify a maintenance complaint into a category.
    
    Algorithm:
        1. Preprocess the text (lowercase, remove punctuation)
        2. Score the text against each category's keyword dictionary
        3. Normalize scores to get confidence values
        4. Select the highest-scoring category
        5. If no category exceeds the threshold, classify as "Unknown"
    
    Args:
        complaint: Raw natural language maintenance complaint text
        
    Returns:
        ClassificationResult with category, confidence, and matched keywords
    """
    processed = _preprocess_text(complaint)

    scores: dict[str, float] = {}
    all_matched: dict[str, list[str]] = {}

    for category, keywords in CATEGORY_KEYWORDS.items():
        score, matched = _score_category(processed, keywords)
        scores[category] = score
        all_matched[category] = matched

    total_score = sum(scores.values())

    # Normalize scores into confidence percentages
    if total_score > 0:
        normalized = {cat: score / total_score for cat, score in scores.items()}
    else:
        normalized = {cat: 0.0 for cat in scores}

    # Find the winning category
    best_category = max(normalized, key=normalized.get)
    best_confidence = normalized[best_category]

    # If confidence is below threshold or no keywords matched, classify as Unknown
    if best_confidence < MIN_CONFIDENCE_THRESHOLD or total_score == 0:
        return ClassificationResult(
            category="Unknown",
            confidence=0.0,
            keywords_matched=[],
            all_scores=normalized,
        )

    return ClassificationResult(
        category=best_category,
        confidence=best_confidence,
        keywords_matched=all_matched[best_category],
        all_scores=normalized,
    )
