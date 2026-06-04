"""
Model training and evaluation utilities
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, auc
from imblearn.over_sampling import SMOTE


def train_test_split_stratified(X, y, test_size=0.2, random_state=42):
    """
    Perform stratified train-test split
    
    Parameters:
    -----------
    X : DataFrame or array
        Features
    y : Series or array
        Target variable
    test_size : float
        Proportion for test set
    random_state : int
        Random seed for reproducibility
    
    Returns:
    --------
    X_train, X_test, y_train, y_test
    """
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def apply_smote(X_train, y_train, random_state=42):
    """
    Apply SMOTE to balance classes
    
    Parameters:
    -----------
    X_train : DataFrame or array
        Training features
    y_train : Series or array
        Training target
    random_state : int
        Random seed
    
    Returns:
    --------
    X_resampled, y_resampled
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"Before SMOTE: Class 0: {(y_train == 0).sum()}, Class 1: {(y_train == 1).sum()}")
    print(f"After SMOTE: Class 0: {(y_resampled == 0).sum()}, Class 1: {(y_resampled == 1).sum()}")
    
    return X_resampled, y_resampled


def train_logistic_regression(X_train, y_train, class_weight='balanced', max_iter=1000, random_state=42):
    """
    Train Logistic Regression model
    """
    model = LogisticRegression(
        class_weight=class_weight,
        max_iter=max_iter,
        random_state=random_state
    )
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=10, class_weight='balanced', random_state=42):
    """
    Train Random Forest model
    """
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        class_weight=class_weight,
        random_state=random_state,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, scale_pos_weight=None, random_state=42):
    """
    Train XGBoost model
    """
    if scale_pos_weight is None:
        scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
    
    model = XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        eval_metric='logloss',
        use_label_encoder=False
    )
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test):
    """
    Evaluate model with multiple metrics
    
    Returns:
    --------
    Dictionary with precision, recall, f1, auc_pr, confusion_matrix
    """
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # AUC-PR
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
    au_prc = auc(recall_curve, precision_curve)
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'au_prc': au_prc,
        'confusion_matrix': cm,
        'predictions': y_pred,
        'probabilities': y_pred_proba
    }

def cross_validate_model(model, X, y, cv=5, scoring='f1'):
    """
    Perform cross-validation
    """
    skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring)
    
    print(f"Cross-validation {scoring} scores: {scores}")
    print(f"Mean: {scores.mean():.4f} (+/- {scores.std():.4f})")
    
    return scores


def save_model(model, filepath):
    """
    Save model to disk
    """
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filepath):
    """
    Load model from disk
    """
    return joblib.load(filepath)