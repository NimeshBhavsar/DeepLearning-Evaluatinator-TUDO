"""
Submission: Noisy Linear Model (Intentionally ~60% accurate)
============================================================
Uses only x feature and adds noise to predictions.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    X_train = train_df[['x']]  # Only use x, ignore y
    y_train = train_df['z']
    X_test = test_df[['x']]
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    # Add noise to make it ~60% accurate
    np.random.seed(42)
    noise = np.random.normal(0, 0.3, size=predictions.shape)
    noisy_predictions = predictions + noise
    
    return noisy_predictions
