"""
Sample Submission Template
==========================

This file demonstrates how to create a valid submission for the Evaluatinator.

Instructions:
1. Copy this file to the ./submissions folder
2. Rename it (e.g., my_model.py)
3. Implement your prediction logic in the predict() function
4. Run: python evaluator.py

Your submission MUST contain a function named 'predict' with the following signature:
    predict(train_df, test_df) -> predictions

Note: This sample file is excluded from evaluation when in the root directory.
"""

import numpy as np
import pandas as pd


def predict(train_df: pd.DataFrame, test_df: pd.DataFrame) -> np.ndarray:
    """
    Train a model on training data and make predictions on test data.
    
    This is the REQUIRED function that the evaluator will call.
    
    Args:
        train_df: pandas DataFrame containing the training data.
                  Includes all feature columns AND the target column 'z'.
                  Example columns: ['feature1', 'feature2', ..., 'z']
        
        test_df: pandas DataFrame containing the test data.
                 Includes all feature columns but NO target column.
                 Example columns: ['feature1', 'feature2', ...]
    
    Returns:
        predictions: Predicted values for the test data.
                     Can be: numpy array, pandas Series, or Python list.
                     Length MUST match len(test_df).
    
    Example:
        >>> train_df = pd.DataFrame({'x': [1, 2, 3], 'z': [0, 0, 1]})
        >>> test_df = pd.DataFrame({'x': [1.5, 2.5]})
        >>> predictions = predict(train_df, test_df)
        >>> print(predictions)
        [0, 1]
    """
    
    # =========================================================================
    # EXAMPLE 1: Simple Threshold Classifier
    # =========================================================================
    # This is a basic example - replace with your own model!
    
    # Separate features and target from training data
    target_column = 'z'
    feature_columns = [col for col in train_df.columns if col != target_column]
    
    X_train = train_df[feature_columns]
    y_train = train_df[target_column]
    X_test = test_df[feature_columns]
    
    # Simple approach: use mean of first feature as threshold
    threshold = X_train.iloc[:, 0].mean()
    predictions = (X_test.iloc[:, 0] > threshold).astype(int).values
    
    return predictions


# =============================================================================
# ALTERNATIVE EXAMPLES (uncomment to use)
# =============================================================================

# def predict(train_df, test_df):
#     """Example using scikit-learn Decision Tree."""
#     from sklearn.tree import DecisionTreeClassifier
#     
#     X_train = train_df.drop('z', axis=1)
#     y_train = train_df['z']
#     X_test = test_df
#     
#     model = DecisionTreeClassifier(max_depth=5, random_state=42)
#     model.fit(X_train, y_train)
#     
#     return model.predict(X_test)


# def predict(train_df, test_df):
#     """Example using scikit-learn Random Forest."""
#     from sklearn.ensemble import RandomForestClassifier
#     
#     X_train = train_df.drop('z', axis=1)
#     y_train = train_df['z']
#     X_test = test_df
#     
#     model = RandomForestClassifier(n_estimators=100, random_state=42)
#     model.fit(X_train, y_train)
#     
#     return model.predict(X_test)


# def predict(train_df, test_df):
#     """Example using scikit-learn Logistic Regression."""
#     from sklearn.linear_model import LogisticRegression
#     from sklearn.preprocessing import StandardScaler
#     
#     X_train = train_df.drop('z', axis=1)
#     y_train = train_df['z']
#     X_test = test_df
#     
#     # Scale features
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
#     
#     model = LogisticRegression(random_state=42)
#     model.fit(X_train_scaled, y_train)
#     
#     return model.predict(X_test_scaled)


# def predict(train_df, test_df):
#     """Example for regression task using Linear Regression."""
#     from sklearn.linear_model import LinearRegression
#     
#     X_train = train_df.drop('z', axis=1)
#     y_train = train_df['z']
#     X_test = test_df
#     
#     model = LinearRegression()
#     model.fit(X_train, y_train)
#     
#     return model.predict(X_test)


# =============================================================================
# Testing (only runs when executed directly, not when imported)
# =============================================================================

if __name__ == "__main__":
    # Create simple test data
    print("Testing sample submission...")
    
    train_data = pd.DataFrame({
        'feature1': [1.0, 2.0, 3.0, 4.0, 5.0],
        'feature2': [2.0, 3.0, 4.0, 5.0, 6.0],
        'z': [0, 0, 1, 1, 1]
    })
    
    test_data = pd.DataFrame({
        'feature1': [1.5, 3.5, 4.5],
        'feature2': [2.5, 4.5, 5.5]
    })
    
    predictions = predict(train_data, test_data)
    
    print(f"Training samples: {len(train_data)}")
    print(f"Test samples: {len(test_data)}")
    print(f"Predictions: {predictions}")
    print(f"Prediction type: {type(predictions)}")
    print("\n✓ Sample submission working correctly!")
