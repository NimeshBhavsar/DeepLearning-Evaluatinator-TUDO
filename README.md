# 🎯 Evaluatinator

A modular Python tool for automated evaluation of machine learning submissions. Evaluates multiple `.py` submission files, computes metrics, and generates professional PDF reports.

## Features

- ✅ **Automatic Evaluation** — Dynamically imports and executes `predict()` functions
- 📊 **Comprehensive Metrics** — Accuracy, Precision, Recall, F1 Score, MSE
- 📈 **Visualizations** — Confusion matrices, scatter plots, residual plots
- 📄 **PDF Reports** — Professional reports for each submission
- 🛡️ **Error Handling** — Graceful handling of missing functions, invalid outputs
- 🔄 **Multi-Submission** — Evaluate multiple submissions in one run

## Quick Start

### 1. Clone the Repository

```bash
git clone https://github.com/sheryl-bellary/evaluatinator.git
cd evaluatinator
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Add Your Files

Place your files in the `./submissions` folder:

```
submissions/
├── train.csv           # Training data with target 'z'
├── test.csv            # Test features (no target)
├── test_internal.csv   # Test data with ground truth 'z'
├── my_model.py         # Your submission file
└── another_model.py    # Additional submissions (optional)
```

### 4. Run the Evaluator

```bash
python evaluator.py
```

### 5. View Results

Reports are saved to `./submissions/reports/`:
- `report_my_model.pdf`
- `report_another_model.pdf`

## Submission Format

Each submission file must contain a `predict` function:

```python
def predict(train_df, test_df):
    """
    Train a model and return predictions.
    
    Args:
        train_df: pandas DataFrame with training data (includes 'z' column)
        test_df: pandas DataFrame with test features (no 'z' column)
    
    Returns:
        Predictions as numpy array, pandas Series, or list
    """
    # Example: Simple threshold classifier
    from sklearn.tree import DecisionTreeClassifier
    
    X_train = train_df.drop('z', axis=1)
    y_train = train_df['z']
    
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    
    predictions = model.predict(test_df)
    return predictions
```

See `sample_submission.py` for a complete example.

## Data Format

### train.csv
```csv
feature1,feature2,...,z
1.0,2.0,...,0
2.0,3.0,...,1
```

### test.csv
```csv
feature1,feature2,...
1.5,2.5,...
2.5,3.5,...
```

### test_internal.csv
```csv
feature1,feature2,...,z
1.5,2.5,...,0
2.5,3.5,...,1
```

## Command Line Options

```bash
# Use default ./submissions folder
python evaluator.py

# Use custom folder
python evaluator.py --folder /path/to/submissions

# Short form
python evaluator.py -f ./my_data
```

## Output

### Console Output
```
======================================================================
🎯 Evaluatinator - ML Submission Evaluator
======================================================================

📁 Submissions folder: /path/to/submissions

📂 Loading data files...
  ✓ Loaded train.csv: 1000 rows, 5 columns
  ✓ Loaded test.csv: 200 rows, 4 columns
  ✓ Loaded test_internal.csv: 200 rows, 5 columns

📋 Found 2 submission(s):
    - my_model.py
    - another_model.py

======================================================================
Starting Evaluation
======================================================================

  📝 Evaluating: my_model
    ✓ Loaded predict() function
    ✓ Executed predictions
    ✓ Validated predictions (n=200)
    ✓ Computed metrics (Classification)
      Accuracy: 0.9500, F1: 0.9487
    📄 Report saved: report_my_model.pdf

======================================================================
Evaluation Summary
======================================================================

✓ Successful: 2/2
✗ Failed: 0/2

📊 Results Overview:

Submission                  Accuracy  Precision     Recall         F1          MSE
-----------------------------------------------------------------------------
my_model                      0.9500     0.9512     0.9463     0.9487     0.050000
another_model                 0.8800     0.8756     0.8842     0.8799     0.120000

📁 Reports saved to: ./submissions/reports

✅ Evaluation complete!
```

### PDF Report Contents
- Submission name and timestamp
- Task type (Classification/Regression)
- Metrics table
- Confusion matrix (for classification)
- Prediction analysis plots

## Project Structure

```
evaluatinator/
├── evaluator.py           # Main evaluation script
├── sample_submission.py   # Example submission template
├── requirements.txt       # Python dependencies
├── README.md              # This file
├── .gitignore             # Git ignore rules
└── submissions/           # Place your files here
    ├── README.md          # Instructions for submissions
    ├── .gitkeep           # Keeps folder in git
    └── reports/           # Generated reports (auto-created)
```

## Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- reportlab

## License

MIT License - feel free to use and modify.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## Troubleshooting

### "No .py submission files found"
Make sure your submission files are in `./submissions/` and have a `.py` extension.

### "Function 'predict' not found"
Your submission file must contain a function named exactly `predict`.

### "predict() returned X predictions, but expected Y"
Your predictions must match the number of rows in `test.csv`.

### Missing CSV files
Ensure `train.csv`, `test.csv`, and `test_internal.csv` are in the submissions folder.
