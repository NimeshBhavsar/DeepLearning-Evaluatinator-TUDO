"""
Submission: Gradient Boosting Regressor
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


def predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    X_train = train_df[['x', 'y']]
    y_train = train_df['z']
    X_test = test_df[['x', 'y']]
    
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    return model.predict(X_test)
