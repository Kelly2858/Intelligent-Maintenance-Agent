# Assumptions, Trade-offs & Production Improvements

## Assumptions Made

1. **Self-contained system**: The application runs entirely locally with no external API keys required. SQLite is used for zero-configuration database setup.

2. **Rule-based classification is sufficient**: For this assessment, a weighted keyword-matching algorithm provides deterministic, explainable results. The architecture is designed so the classifier module can be swapped with an LLM-based classifier without changing any other code.

3. **English-only input**: Complaints are assumed to be in English. The text preprocessor lowercases and strips special characters.

4. **Single-user operation**: No authentication or multi-tenancy is implemented. The system processes complaints from a single operator or team.

5. **Categories are exhaustive**: The four categories (Electrical, Mechanical, Sensor, Unknown) cover the scope of industrial maintenance. "Unknown" acts as the catch-all.

6. **Priority is inferred from text**: Priority is derived from urgency signals in the complaint text combined with category context (e.g., Electrical issues get a safety bias multiplier).

---

## Trade-offs Considered

| Decision | Chosen Approach | Alternative | Reasoning |
|---|---|---|---|
| **Classification method** | Weighted keyword matching | LLM API (GPT-4, Gemini) | Deterministic, zero-cost, no API dependency, fully explainable. An LLM would handle ambiguous cases better but adds latency, cost, and non-determinism. |
| **Database** | SQLite | PostgreSQL / MySQL | Zero configuration, single-file database, perfect for assessment scope. Production would use PostgreSQL. |
| **Web framework** | FastAPI | Flask / Django | FastAPI provides automatic OpenAPI docs, async support, Pydantic validation, and type hints out of the box. |
| **Frontend** | Vanilla HTML/CSS/JS | React / Vue | No build step required, instantly runnable, demonstrates clean code without framework overhead. |
| **Priority assignment** | Keyword + category bias | ML model / LLM | Transparent scoring with explainable reasoning. ML would require labeled training data we don't have. |
| **Confidence scoring** | Normalized keyword weights | Probability from ML model | Provides a meaningful 0-1 score showing relative category strength without requiring model training. |

---

## Improvements for Production

### 1. LLM-Powered Classification
Replace the keyword classifier with a fine-tuned model or LLM API call using structured output (JSON mode). This would handle ambiguous, poorly-worded, or multi-issue complaints far better than keyword matching.

### 2. PostgreSQL + Connection Pooling
Migrate from SQLite to PostgreSQL with connection pooling (e.g., asyncpg) for concurrent multi-user access and proper ACID guarantees under load.

### 3. Authentication & RBAC
Add JWT-based authentication with role-based access control. Maintenance technicians would submit, supervisors would triage, and admins would manage the system.

### 4. Real-time Notifications
Integrate WebSocket or SSE for real-time dashboard updates. High-priority tickets would trigger immediate push notifications to on-call engineers.

### 5. Historical ML Model
Train a supervised classifier on accumulated ticket data. As the database grows, labeled historical data becomes a valuable training asset for a fine-tuned BERT or similar model.

### 6. Multi-language Support
Add translation middleware (e.g., Google Translate API) to handle complaints in multiple languages, critical for global industrial operations.

### 7. Integration APIs
Connect to existing CMMS (Computerized Maintenance Management Systems) like SAP PM, IBM Maximo, or ServiceNow for automated work order creation.

### 8. Audit Trail & Logging
Add structured logging (JSON logs), request tracing, and a full audit trail for compliance in regulated industries.

### 9. Containerization
Package with Docker and docker-compose for consistent deployment across environments. Add CI/CD pipeline with automated testing.

### 10. Monitoring & Observability
Add Prometheus metrics, Grafana dashboards, and alerting for classifier accuracy drift detection over time.
