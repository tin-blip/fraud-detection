"""
Unit tests for model training and evaluation
"""

import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

def test_model_training():
    """Test that models can train on dummy data"""
    # Create dummy data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train models
    lr = LogisticRegression(max_iter=100)
    lr.fit(X_train, y_train)
    
    rf = RandomForestClassifier(n_estimators=10, random_state=42)
    rf.fit(X_train, y_train)
    
    # Assert models have coefficients/trees
    assert lr.coef_ is not None
    assert rf.n_estimators == 10
    
    print("✓ Model training test passed")


def test_model_prediction():
    """Test that models can make predictions"""
    X = np.random.randn(20, 3)
    y = np.random.randint(0, 2, 20)
    
    model = LogisticRegression(max_iter=100)
    model.fit(X, y)
    
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    assert len(predictions) == 20
    assert probabilities.shape == (20, 2)
    assert np.all((predictions == 0) | (predictions == 1))
    
    print("✓ Model prediction test passed")


def test_imbalance_handling():
    """Test that class imbalance is handled"""
    # Create imbalanced data (90% class 0, 10% class 1)
    X = np.random.randn(100, 2)
    y = np.array([0] * 90 + [1] * 10)
    
    # Use class_weight='balanced'
    model = LogisticRegression(class_weight='balanced', max_iter=100)
    model.fit(X, y)
    
    # Should still predict some fraud
    predictions = model.predict(X)
    assert np.sum(predictions == 1) > 0
    
    print("✓ Imbalance handling test passed")


def test_evaluation_metrics():
    """Test that evaluation metrics work correctly"""
    # Perfect predictions
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    assert accuracy == 1.0
    assert f1 == 1.0
    
    # Bad predictions
    y_pred_bad = np.array([1, 1, 0, 0])
    accuracy_bad = accuracy_score(y_true, y_pred_bad)
    f1_bad = f1_score(y_true, y_pred_bad)
    
    assert accuracy_bad == 0.0
    assert f1_bad == 0.0
    
    print("✓ Evaluation metrics test passed")


def test_train_test_split_stratified():
    """Test that stratified split preserves class distribution"""
    y = np.array([0] * 80 + [1] * 20)
    
    from sklearn.model_selection import train_test_split
    _, y_train, _, y_test = train_test_split(y, y, test_size=0.2, stratify=y, random_state=42)
    
    # Check distribution preserved
    assert abs(y_train.mean() - y.mean()) < 0.01
    
    print("✓ Stratified split test passed")


if __name__ == "__main__":
    test_model_training()
    test_model_prediction()
    test_imbalance_handling()
    test_evaluation_metrics()
    test_train_test_split_stratified()
    print("\n✅ All model tests passed!")