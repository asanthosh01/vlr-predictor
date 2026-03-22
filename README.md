# Fantasy Valorant Predictor

Predict outcomes of professional Valorant matches using real data from [VLR.gg](https://vlr.gg)!  
Built end-to-end with Python: data collection, feature engineering, model training, and prediction.

---

## Project Overview
The **Fantasy Valorant Predictor** is a machine learning project that forecasts which team will win an upcoming professional match.  
It uses scraped match data from VLR.gg, team rankings, and player statistics to train an XGBoost classifier.

---

## Tech Stack
- **Python 3.10+**
- **Pandas**, **NumPy** – Data wrangling  
- **Scikit-learn**, **XGBoost** – Model training  
- **Requests**, **BeautifulSoup** – Web scraping  
- **Flask / FastAPI** – Prediction API (for deployment)  
- **Render / Railway** – Cloud hosting  

---

## How It Works
1. **Data Collection** → `data_collector_basic.py` scrapes match, ranking, and player data from VLR.gg’s API.  
2. **Feature Engineering** → `feature_engineering.py` converts raw data into model-ready numeric features.  
3. **Model Training** → Train an XGBoost classifier on historical match results.  
4. **Prediction Interface** → A simple web app lets users select two teams and see win probabilities.

---

## Model Performance (Baseline)
| Metric | Value |
|---------|-------|
| Accuracy | ~0.68 |
| Precision | 0.70 |
| Recall | 0.66 |
| ROC-AUC | 0.72 |

*(Results based on current dataset of NA matches — will improve as more features are added!)*

---

## How to Run Locally
```bash
# Clone the repo
git clone https://github.com/<your-username>/fantasy-valorant-predictor.git
cd fantasy-valorant-predictor

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt

# Run data pipeline
python src/feature_engineering.py

# Train model
python src/train_model.py

# Run web app
python app.py
