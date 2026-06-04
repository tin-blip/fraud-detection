import pytest
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

def test_model_training():
    """Test that a model can be trained"""
    # Create dummy data
    X = np.random.randn(100, 5)
    y = np.random.randint(0, 2, 100)
    
    model = LogisticRegression(max_iter=100)
    model.fit(X, y)
    
    assert model.coef_ is not None
    print("✓ Model training test passed")

def test_prediction():
    """Test that model makes predictions"""
    X = np.random.randn(10, 5)
    model = LogisticRegression()
    model.fit(X[:5], np.array([0,0,0,1,1]))
    
    preds = model.predict(X[5:])
    assert len(preds) == 5
    print("✓ Prediction test passed")

if name == "main":
    test_model_training()
    test_prediction()