import sys
import os

# Ensure the correct path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from database.db import SessionLocal, init_db
from database.models import MaintenanceTicket
from agent.maintenance_agent import MaintenanceAgent
from agent.ml_trainer import train_agent_models

def seed_db():
    db = SessionLocal()
    
    # Diverse complaints covering the requested examples
    complaints = [
        # --- Electrical ---
        "The main transformer is overheating and sparking.",
        "Severe short circuit detected in the primary power supply unit.",
        "Circuit breaker keeps tripping randomly on assembly line 4.",
        "There is a burning smell coming from the main electrical panel.",
        "Voltage fluctuations observed, causing the inverter to shut down.",
        "Ground fault alarm triggered in the switchgear cabinet.",
        "Motor winding shows signs of insulation failure and electrical shorts.",
        "Complete power outage in the control room due to a blown high-amp fuse.",
        "Dangerous arcing noticed near the high voltage capacitor bank.",
        "Three-phase motor is missing one phase and humming very loudly.",
        
        # --- Mechanical ---
        "Conveyor belt bearing is completely worn out and making a loud grinding noise.",
        "Gearbox on the main drive shaft is vibrating excessively.",
        "Hydraulic pump is leaking fluid heavily near the primary seal.",
        "Drive belt snapped on the secondary cooling fan.",
        "Shaft misalignment is causing severe friction, heat, and wear.",
        "Pneumatic cylinder is stuck and won't actuate properly.",
        "Metal fatigue and cracking observed on the turbine blades.",
        "Lubrication failed on the main sprocket chain, causing extreme friction.",
        "Safety valve is jammed open and mechanical override isn't working.",
        "Drive roller is deformed, bent, and causing the conveyor to jam.",
        
        # --- Sensor ---
        "Temperature sensor on reactor 3 is giving false readings and needs immediate recalibration.",
        "Pressure gauge is showing highly erratic measurements outside normal bounds.",
        "Flow meter signal is lost intermittently, causing false alarms.",
        "Proximity sensor is covered in dust and failing to detect objects on the line.",
        "Thermocouple has severe sensor drift over the last 48 hours.",
        "Liquid level sensor is stuck at 100% despite the storage tank being empty.",
        "Speed encoder on the spindle is sending highly inaccurate RPM data.",
        "Laser alignment sensor requires a complete zero-point recalibration.",
        "Vibration sensor is returning out of range values even when machine is off.",
        "Photoelectric safety detector has a weak signal and needs realignment."
    ]
    
    # Load complaints from sample_inputs_outputs.json
    samples_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample_inputs_outputs.json")
    if os.path.exists(samples_path):
        import json
        with open(samples_path, "r", encoding="utf-8") as f:
            samples = json.load(f)
            for sample in samples:
                complaint_text = sample.get("input", {}).get("complaint")
                if complaint_text and complaint_text not in complaints:
                    complaints.append(complaint_text)

    # Initialize DB tables if they don't exist
    init_db()
    
    # We temporarily disable ML models to force rule-based processing for generating the ground-truth training set
    agent = MaintenanceAgent(db)
    agent.ml_category = None
    agent.ml_priority = None
    
    print("Processing and storing tickets...")
    for comp in complaints:
        agent.process_complaint(comp)
        
    print(f"Stored {len(complaints)} high-quality training tickets in the database.")
    
    print("Training ML Models...")
    try:
        res = train_agent_models(db)
        print("Success:", res)
    except Exception as e:
        print(f"Failed to train: {e}")
        
    db.close()

if __name__ == "__main__":
    seed_db()
