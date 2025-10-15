"""
Train XGBoost Model for Match Prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class MatchPredictor:
    def __init__(self):
        self.model = None
        self.feature_names = None
        
    def load_data(self, filepath='data/processed/training_data_full.csv'):
        """Load processed features"""
        df = pd.read_csv(filepath)
        
        # Separate features and target
        X = df.drop('team1_won', axis=1)
        y = df['team1_won']
        
        self.feature_names = list(X.columns)
        
        return X, y
    
    def train(self, X, y, test_size=0.2, random_state=42):
        """Train XGBoost model"""
        print("=" * 60)
        print("TRAINING XGBOOST MODEL")
        print("=" * 60)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"\nTraining samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        print(f"Features: {len(self.feature_names)}")
        
        # Train XGBoost
        self.model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.1,
            random_state=random_state,
            eval_metric='logloss'
        )
        
        print("\nTraining model...")
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Evaluate
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        print("\n" + "=" * 60)
        print("MODEL PERFORMANCE")
        print("=" * 60)
        print(f"Training Accuracy: {train_acc*100:.2f}%")
        print(f"Testing Accuracy: {test_acc*100:.2f}%")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='accuracy')
        print(f"\n5-Fold CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
        
        print("\n" + "=" * 60)
        print("CLASSIFICATION REPORT (Test Set)")
        print("=" * 60)
        print(classification_report(y_test, y_pred_test, 
                                   target_names=['Team 2 Wins', 'Team 1 Wins']))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_test)
        print("\nConfusion Matrix:")
        print(cm)
        
        # Feature importance
        self.plot_feature_importance()
        
        return X_train, X_test, y_train, y_test, y_pred_test
    
    def plot_feature_importance(self):
        """Plot feature importance"""
        importance = self.model.feature_importances_
        feature_importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance_df['feature'], feature_importance_df['importance'])
        plt.xlabel('Importance')
        plt.title('Feature Importance in Match Prediction')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
        print("\n Feature importance plot saved to models/feature_importance.png")
        plt.close()
    
    def save_model(self, filepath='models/match_predictor.joblib'):
        """Save trained model"""
        joblib.dump(self.model, filepath)
        print(f"\n Model saved to {filepath}")
    
    def predict(self, team1_features, team2_features):
        """Predict match outcome"""
        # Create feature vector
        features = {
            'team1_rank': team1_features['rank'],
            'team1_win_rate': team1_features['win_rate'],
            'team1_total_games': team1_features['total_games'],
            'team1_wins': team1_features['wins'],
            'team2_rank': team2_features['rank'],
            'team2_win_rate': team2_features['win_rate'],
            'team2_total_games': team2_features['total_games'],
            'team2_wins': team2_features['wins'],
            'rank_difference': team1_features['rank'] - team2_features['rank'],
            'win_rate_difference': team1_features['win_rate'] - team2_features['win_rate'],
            'experience_difference': team1_features['total_games'] - team2_features['total_games']
        }
        
        X = pd.DataFrame([features])
        prediction = self.model.predict_proba(X)[0]
        
        return {
            'team1_win_prob': prediction[1] * 100,
            'team2_win_prob': prediction[0] * 100
        }

if __name__ == "__main__":
    # Initialize predictor
    predictor = MatchPredictor()
    
    # Load data
    X, y = predictor.load_data()
    
    # Train model
    predictor.train(X, y)
    
    # Save model
    predictor.save_model()
    
    print("\n" + "=" * 60)
    print("✓ TRAINING COMPLETE!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Check models/feature_importance.png")
    print("2. Model saved and ready for predictions")
    print("3. Ready to build web interface!")