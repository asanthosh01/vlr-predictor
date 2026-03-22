"""
Combined Model Trainer - Merge all features and train improved model
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import joblib
import matplotlib.pyplot as plt
from feature_engineering import EnhancedFeatureEngineer

class CombinedModelTrainer:
    def __init__(self):
        self.model = None
        self.feature_names = None
        
    def merge_features(self):
        """Merge advanced features with ranking features"""
        print("=" * 70)
        print("MERGING ALL FEATURE SOURCES")
        print("=" * 70)
        
        # Load advanced features
        advanced = pd.read_csv('data/processed/advanced_features.csv')
        print(f"✓ Loaded {len(advanced)} matches with advanced features")
        
        # Load rankings for lookup
        engineer = EnhancedFeatureEngineer()
        engineer.load_latest_data()
        
        # Add ranking features to each match
        ranking_features = []
        
        for idx, row in advanced.iterrows():
            if idx % 100 == 0:
                print(f"  Adding rankings for match {idx}/{len(advanced)}...")
            
            t1_feat = engineer.get_team_features(row['team1'])
            t2_feat = engineer.get_team_features(row['team2'])
            
            features = {
                'team1_rank': t1_feat['rank'],
                'team1_win_rate': t1_feat['win_rate'],
                'team1_total_games': t1_feat['total_games'],
                'team1_wins': t1_feat['wins'],
                'team2_rank': t2_feat['rank'],
                'team2_win_rate': t2_feat['win_rate'],
                'team2_total_games': t2_feat['total_games'],
                'team2_wins': t2_feat['wins'],
                'rank_difference': t1_feat['rank'] - t2_feat['rank'],
                'win_rate_difference': t1_feat['win_rate'] - t2_feat['win_rate'],
                'experience_difference': t1_feat['total_games'] - t2_feat['total_games']
            }
            
            ranking_features.append(features)
        
        ranking_df = pd.DataFrame(ranking_features)
        
        # Combine both feature sets
        combined = pd.concat([
            ranking_df.reset_index(drop=True),
            advanced[['team1_recent_win_rate', 'team1_recent_wins', 'team1_recent_matches',
                     'team2_recent_win_rate', 'team2_recent_wins', 'team2_recent_matches',
                     'h2h_matches_played', 'h2h_team1_wins', 'h2h_team1_win_rate',
                     'recent_form_difference', 'team1_won']].reset_index(drop=True)
        ], axis=1)
        
        print(f"\n✓ Combined dataset: {len(combined)} matches")
        print(f"✓ Total features: {len(combined.columns) - 1}")
        
        return combined
    
    def train_model(self, X, y):
        """Train XGBoost with all features"""
        print("\n" + "=" * 70)
        print("TRAINING IMPROVED XGBOOST MODEL")
        print("=" * 70)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining samples: {len(X_train)}")
        print(f"Testing samples: {len(X_test)}")
        print(f"Features: {len(self.feature_names)}")
        
        # Train XGBoost
        self.model = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
            eval_metric='logloss',
            subsample=0.8,
            colsample_bytree=0.8
        )
        
        print("\nTraining model...")
        self.model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = self.model.predict(X_train)
        y_pred_test = self.model.predict(X_test)
        
        # Evaluate
        train_acc = accuracy_score(y_train, y_pred_train)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        print("\n" + "=" * 70)
        print("MODEL PERFORMANCE")
        print("=" * 70)
        print(f"Training Accuracy: {train_acc*100:.2f}%")
        print(f"Testing Accuracy: {test_acc*100:.2f}%")
        
        # Cross-validation
        cv_scores = cross_val_score(self.model, X, y, cv=5, scoring='accuracy')
        print(f"\n5-Fold CV Accuracy: {cv_scores.mean()*100:.2f}% (+/- {cv_scores.std()*100:.2f}%)")
        
        print("\n" + "=" * 70)
        print("CLASSIFICATION REPORT (Test Set)")
        print("=" * 70)
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
        
        plt.figure(figsize=(12, 8))
        top_features = feature_importance_df.head(15)
        plt.barh(top_features['feature'], top_features['importance'])
        plt.xlabel('Importance')
        plt.title('Top 15 Feature Importance - Combined Model')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('models/feature_importance_combined.png', dpi=300, bbox_inches='tight')
        print("\n✓ Feature importance plot saved to models/feature_importance_combined.png")
        plt.close()
    
    def save_model(self, filepath='models/match_predictor_v2.joblib'):
        """Save trained model"""
        joblib.dump(self.model, filepath)
        print(f"\n✓ Model saved to {filepath}")

if __name__ == "__main__":
    trainer = CombinedModelTrainer()
    
    # Merge all features
    combined_data = trainer.merge_features()
    
    # Prepare X and y
    X = combined_data.drop('team1_won', axis=1)
    y = combined_data['team1_won']
    
    trainer.feature_names = list(X.columns)
    
    # Train model
    trainer.train_model(X, y)
    
    # Save
    trainer.save_model()
    
    print("\n" + "=" * 70)
    print("✓ TRAINING COMPLETE - MODEL V2!")
    print("=" * 70)
    print("\nImprovements over V1:")
    print("  • Added recent form features (last 5 matches)")
    print("  • Added head-to-head history")
    print("  • Increased model complexity (150 trees, depth 4)")
    print("=" * 70)