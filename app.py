"""Flask app that remains usable when trained artifacts are absent."""
import json
import os
from pathlib import Path
import pandas as pd
from flask import Flask, jsonify, render_template, request
from xgboost import XGBClassifier
from src.chronological import FEATURES, build_features

ROOT = Path(__file__).resolve().parent

def create_app(data_path=None, artifact_dir=None):
    app = Flask(__name__)
    data_path = Path(data_path or os.environ.get("MATCH_DATA", ROOT / "data/raw/matches.csv"))
    artifacts = Path(artifact_dir or os.environ.get("MODEL_DIR", ROOT / "artifacts"))
    model, state, report, teams, unavailable = None, None, None, [], None
    try:
        report = json.loads((artifacts / "evaluation.json").read_text())
        if report["features"] != FEATURES:
            raise ValueError("Model feature schema is incompatible; retrain")
        matches = pd.read_csv(data_path)
        import hashlib
        if hashlib.sha256(data_path.read_bytes()).hexdigest() != report["data_sha256"]:
            raise ValueError("Dataset differs from evaluated artifacts; retrain")
        _, state = build_features(matches)
        teams = sorted(set(matches.team1) | set(matches.team2))
        model = XGBClassifier()
        model.load_model(artifacts / "model.json")
    except (OSError, ValueError, KeyError) as exc:
        unavailable = f"Model unavailable: {exc}. Run python -m src.collect_history, then python -m src.evaluate."

    @app.get("/")
    def home():
        return render_template("index.html", teams=teams, report=report, error=unavailable)

    @app.get("/health")
    def health():
        return jsonify({"ready": model is not None, "error": unavailable})

    @app.post("/predict")
    def predict():
        if model is None:
            return jsonify({"error": unavailable}), 503
        body = request.get_json(silent=True)
        if not isinstance(body, dict):
            return jsonify({"error": "JSON object required"}), 400
        a, b = body.get("team1"), body.get("team2")
        if not isinstance(a, str) or not isinstance(b, str) or a not in teams or b not in teams or a == b:
            return jsonify({"error": "Choose two different known teams"}), 400
        vector = state.features(a, b)
        probability = float(model.predict_proba(pd.DataFrame([vector], columns=FEATURES))[0, 1])
        return jsonify({"team1": a, "team2": b, "team1_probability": probability,
            "team2_probability": 1-probability, "features": vector,
            "history_through": report["splits"]["test"]["end"],
            "note": "Uncalibrated estimate using the saved historical dataset; not a live-data service."})
    return app

app = create_app()
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=int(os.environ.get("PORT", "5000")), debug=False)
