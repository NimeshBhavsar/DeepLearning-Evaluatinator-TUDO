# Submissions Folder

Place your files here for evaluation.

## Required Files

| File | Description |
|------|-------------|
| `train.csv` | Training data with features and target column `z` |
| `test.csv` | Test data with features only (no target) |
| `test_internal.csv` | Test data with ground truth `z` column for evaluation |
| `*.py` | Your submission file(s) with `predict()` function |

## Submission Format

Each `.py` submission file must contain a function named `predict`:

```python
def predict(train_df, test_df):
    """
    Train a model and make predictions.
    
    Args:
        train_df: pandas DataFrame with training data (includes 'z' column)
        test_df: pandas DataFrame with test features (no 'z' column)
    
    Returns:
        Predictions as numpy array, pandas Series, or list
    """
    # Your code here
    return predictions
```

## Example

See `sample_submission.py` in the root directory for a complete example.

## Running the Evaluator

From the repository root:

```bash
python evaluator.py
```

Reports will be generated in `./submissions/reports/`
