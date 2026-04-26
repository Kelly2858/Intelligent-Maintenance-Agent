import os
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from database.models import MaintenanceTicket

def train_agent_models(db):
    """
    Train Machine Learning models (Category & Priority) using the ticket history
    in the database. Replaces the rule-based approach once trained.
    """
    tickets = db.query(MaintenanceTicket).all()
    if len(tickets) < 3:
        raise ValueError("Need at least 3 tickets in history to train the ML models.")
        
    X = [t.complaint for t in tickets]
    y_cat = [t.category for t in tickets]
    y_pri = [t.priority for t in tickets]
    
    # Create text classification pipelines
    # Using TF-IDF + Random Forest for robust performance on small datasets
    cat_model = make_pipeline(TfidfVectorizer(stop_words='english'), RandomForestClassifier(n_estimators=50, random_state=42))
    cat_model.fit(X, y_cat)
    
    pri_model = make_pipeline(TfidfVectorizer(stop_words='english'), RandomForestClassifier(n_estimators=50, random_state=42))
    pri_model.fit(X, y_pri)
    
    # Ensure models directory exists
    model_dir = os.path.join(os.path.dirname(__file__), "models")
    os.makedirs(model_dir, exist_ok=True)
    
    # Save the trained models
    joblib.dump(cat_model, os.path.join(model_dir, "category_model.pkl"))
    joblib.dump(pri_model, os.path.join(model_dir, "priority_model.pkl"))
    
    return {"message": "Models trained successfully!", "samples": len(X)}
