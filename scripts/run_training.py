"""
Run model training from command line
Usage: python scripts/run_training.py
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(file))))

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE

def main():
    print("=" * 60)
    print("STARTING MODEL TRAINING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load processed data
    print("\n1. Loading processed data...")
    df = pd.read_csv('data/processed/fraud_data_processed.csv')
    print(f"   Loaded {len(df)} rows with {len(df.columns)} columns")
    
    # Step 2: Separate features and target
    print("\n2. Preparing features and target...")
    target_col = 'class'
    feature_cols = [col for col in df.columns if col != target_col]
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Handle categorical columns
    X = pd.get_dummies(X, drop_first=True)
    print(f"   Features: {X.shape[1]}, Target: {y.name}")
    
    # Step 3: Train-test split
    print("\n3. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    print(f"   Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Step 4: Apply SMOTE
    print("\n4. Applying SMOTE to balance classes...")
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    print(f"   Before: {y_train.value_counts().to_dict()}")
    print(f"   After: {y_train_resampled.value_counts().to_dict()}")
    
    # Step 5: Train models
    print("\n5. Training models...")
    
    models = {
        'LogisticRegression': LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        'RandomForest': RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42, n_jobs=-1),
        'XGBoost': XGBClassifier(scale_pos_weight=len(y_train[y_train==0])/len(y_train[y_train==1]), random_state=42)
    }
    
    results = []
    
    for name, model in models.items():
        print(f"\n   Training {name}...")
        model.fit(X_train_resampled, y_train_resampled)
        
        y_pred = model.predict(X_test)
        
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        results.append({
            'model': name,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })
        
        # Save model
        joblib.dump(model, f'models/{name.lower()}_model.pkl')
        print(f"      Saved to models/{name.lower()}_model.pkl")
    
    # Step 6: Display results
    print("\n" + "=" * 60)
    print("TRAINING RESULTS")
    print("=" * 60)
    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    
    # Find best model
    best_model = results_df.loc[results_df['f1'].idxmax(), 'model']
    print(f"\n🏆 BEST MODEL: {best_model} (F1: {results_df['f1'].max():.4f})")
    
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE!")
    print("=" * 60)

if name == "main":
    main()