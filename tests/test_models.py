"""
Unit tests for models
"""

import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

def test_model_training():
    """Test that models can train"""
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    
    lr = LogisticRegression(max_iter=100)
    lr.fit(X, y)
    
    assert lr.coef_ is not None
    print("✓ Model training test passed")


def test_model_prediction():
    """Test predictions"""
    X = np.random.randn(20, 3)
    y = np.random.randint(0, 2, 20)
    
    model = LogisticRegression(max_iter=100)
    model.fit(X, y)
    
    predictions = model.predict(X)
    assert len(predictions) == 20
    print("✓ Prediction test passed")


def test_train_test_split():
    """Test split"""
    y = np.array([0] * 80 + [1] * 20)
    _, y_train, _, y_test = train_test_split(y, y, test_size=0.2, stratify=y, random_state=42)
    assert abs(y_train.mean() - y.mean()) < 0.01
    print("✓ Split test passed")


if name == "main":
    test_model_training()
    test_model_prediction()
    test_train_test_split()
    print("\n✅ All model tests passed!")
