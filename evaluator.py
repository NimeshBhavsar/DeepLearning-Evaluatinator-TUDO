#!/usr/bin/env python3
"""
Evaluatinator - ML Submission Evaluator
=======================================

A modular Python script that automates the evaluation of multiple machine learning
submissions. Each submission is a .py file containing a predict(train_df, test_df)
function. The script generates comprehensive PDF reports with evaluation metrics,
confusion matrices, and visualizations.

Usage:
    python evaluator.py                    # Uses ./submissions folder by default
    python evaluator.py --folder /path/to  # Custom folder path

Expected folder structure:
    ./submissions/
    ├── train.csv           # Training data with features and target 'z'
    ├── test.csv            # Test data with features (no target)
    ├── test_internal.csv   # Test data with ground truth 'z' for evaluation
    ├── submission_1.py     # Submission file with predict() function
    ├── submission_2.py     # Another submission
    └── ...
"""

import os
import sys
import argparse
import importlib.util
import traceback
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for PDF generation
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_squared_error,
    confusion_matrix,
    classification_report
)

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
    PageBreak
)


# =============================================================================
# Configuration
# =============================================================================

# Default folder path - relative to the script location
# This ensures ./submissions is used regardless of where the script is called from
SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_SUBMISSIONS_FOLDER = SCRIPT_DIR / "submissions"

# CSV file names (expected in the submissions folder)
TRAIN_CSV = "train.csv"
TEST_CSV = "test.csv"
TEST_INTERNAL_CSV = "test_internal.csv"

# Target column name
TARGET_COLUMN = "z"

# Report output directory (relative to submissions folder)
REPORTS_SUBDIR = "reports"


# =============================================================================
# Data Classes for Structured Results
# =============================================================================

@dataclass
class DatasetInfo:
    """Container for dataset information."""
    name: str
    folder_path: Path
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    test_internal_df: pd.DataFrame
    ground_truth: np.ndarray


@dataclass
class EvaluationMetrics:
    """Container for all evaluation metrics."""
    accuracy: float
    precision: float
    recall: float
    f1: float
    mse: float
    is_classification: bool
    confusion_mat: Optional[np.ndarray] = None
    class_labels: Optional[List] = None


@dataclass
class EvaluationResult:
    """Container for complete evaluation results of a submission."""
    submission_name: str
    dataset_name: str
    success: bool
    metrics: Optional[EvaluationMetrics] = None
    predictions: Optional[np.ndarray] = None
    ground_truth: Optional[np.ndarray] = None
    test_df: Optional[pd.DataFrame] = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None


# =============================================================================
# Data Loading Functions
# =============================================================================

def load_csv_file(filepath: Path, silent: bool = False) -> pd.DataFrame:
    """
    Load a CSV file into a pandas DataFrame.
    
    Args:
        filepath: Path to the CSV file.
        silent: If True, don't print loading message.
        
    Returns:
        DataFrame containing the CSV data.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        pd.errors.ParserError: If the file cannot be parsed as CSV.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    if not silent:
        print(f"    ✓ Loaded {filepath.name}: {len(df)} rows, {len(df.columns)} columns")
    return df


def is_valid_dataset_folder(folder_path: Path) -> bool:
    """
    Check if a folder contains all required dataset files.
    
    Args:
        folder_path: Path to check.
        
    Returns:
        True if folder contains train.csv, test.csv, and test_internal.csv.
    """
    required_files = [TRAIN_CSV, TEST_CSV, TEST_INTERNAL_CSV]
    return all((folder_path / f).exists() for f in required_files)


def discover_datasets(folder_path: Path) -> List[DatasetInfo]:
    """
    Discover all datasets in the submissions folder.
    
    Supports two modes:
    1. Legacy mode: CSV files directly in submissions folder (single dataset named 'default')
    2. Multi-dataset mode: Subfolders containing CSV files (each subfolder is a dataset)
    
    Args:
        folder_path: Path to the submissions folder.
        
    Returns:
        List of DatasetInfo objects.
    """
    datasets = []
    
    # Check for subfolders with datasets
    for subfolder in folder_path.iterdir():
        if subfolder.is_dir() and not subfolder.name.startswith('.') and subfolder.name != REPORTS_SUBDIR:
            if is_valid_dataset_folder(subfolder):
                datasets.append(subfolder)
    
    # If no dataset subfolders found, check if files are directly in submissions folder (legacy mode)
    if not datasets and is_valid_dataset_folder(folder_path):
        datasets.append(folder_path)
    
    # Load each dataset
    loaded_datasets = []
    for dataset_folder in sorted(datasets, key=lambda p: p.name.lower()):
        dataset_name = "default" if dataset_folder == folder_path else dataset_folder.name
        
        print(f"\n  📁 Loading dataset: {dataset_name}")
        try:
            train_df = load_csv_file(dataset_folder / TRAIN_CSV)
            test_df = load_csv_file(dataset_folder / TEST_CSV)
            test_internal_df = load_csv_file(dataset_folder / TEST_INTERNAL_CSV)
            
            # Validate target column
            if TARGET_COLUMN not in test_internal_df.columns:
                print(f"    ⚠️  Skipping: Target column '{TARGET_COLUMN}' not found")
                continue
            
            ground_truth = test_internal_df[TARGET_COLUMN].values
            
            loaded_datasets.append(DatasetInfo(
                name=dataset_name,
                folder_path=dataset_folder,
                train_df=train_df,
                test_df=test_df,
                test_internal_df=test_internal_df,
                ground_truth=ground_truth
            ))
            print(f"    ✓ Dataset '{dataset_name}' loaded ({len(ground_truth)} test samples)")
            
        except Exception as e:
            print(f"    ⚠️  Error loading dataset '{dataset_name}': {e}")
            continue
    
    return loaded_datasets


def load_all_data(folder_path: Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load all required CSV files from the submissions folder (legacy single-dataset mode).
    
    Args:
        folder_path: Path to the submissions folder.
        
    Returns:
        Tuple of (train_df, test_df, test_internal_df).
        
    Raises:
        FileNotFoundError: If any required CSV file is missing.
    """
    print("\n📂 Loading data files...")
    
    train_df = load_csv_file(folder_path / TRAIN_CSV)
    test_df = load_csv_file(folder_path / TEST_CSV)
    test_internal_df = load_csv_file(folder_path / TEST_INTERNAL_CSV)
    
    # Validate that test_internal has the target column
    if TARGET_COLUMN not in test_internal_df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found in {TEST_INTERNAL_CSV}. "
            f"Available columns: {list(test_internal_df.columns)}"
        )
    
    print(f"  ✓ All data files loaded successfully")
    return train_df, test_df, test_internal_df


# =============================================================================
# Submission Discovery and Loading
# =============================================================================

def discover_submissions(folder_path: Path) -> List[Path]:
    """
    Find all Python submission files in the folder.
    
    Excludes files starting with underscore and special files.
    
    Args:
        folder_path: Path to the submissions folder.
        
    Returns:
        List of paths to submission .py files.
    """
    # Files to exclude from evaluation
    excluded_files = {
        "evaluator.py",
        "sample_submission.py",
        "__init__.py",
    }
    
    submissions = []
    
    for file_path in folder_path.glob("*.py"):
        # Skip private/hidden files
        if file_path.name.startswith("_"):
            continue
        # Skip excluded files
        if file_path.name.lower() in excluded_files:
            continue
            
        submissions.append(file_path)
    
    # Sort for consistent ordering
    submissions.sort(key=lambda p: p.name.lower())
    
    return submissions


def load_predict_function(submission_path: Path) -> callable:
    """
    Dynamically import a submission file and extract the predict() function.
    
    Args:
        submission_path: Path to the .py submission file.
        
    Returns:
        The predict function from the submission.
        
    Raises:
        ImportError: If the module cannot be loaded.
        AttributeError: If the predict function is not found.
    """
    # Create a unique module name to avoid conflicts
    module_name = f"submission_{submission_path.stem}"
    
    # Load the module specification
    spec = importlib.util.spec_from_file_location(module_name, submission_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module specification for {submission_path}")
    
    # Create and execute the module
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    
    try:
        spec.loader.exec_module(module)
    except Exception as e:
        # Clean up the module from sys.modules on failure
        sys.modules.pop(module_name, None)
        raise ImportError(f"Error executing module {submission_path.name}: {e}")
    
    # Extract the predict function
    if not hasattr(module, "predict"):
        sys.modules.pop(module_name, None)
        raise AttributeError(
            f"Function 'predict' not found in {submission_path.name}. "
            f"Available attributes: {[a for a in dir(module) if not a.startswith('_')]}"
        )
    
    predict_func = getattr(module, "predict")
    
    if not callable(predict_func):
        sys.modules.pop(module_name, None)
        raise AttributeError(f"'predict' in {submission_path.name} is not callable")
    
    return predict_func


# =============================================================================
# Prediction Execution and Validation
# =============================================================================

def execute_prediction(
    predict_func: callable,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame
) -> np.ndarray:
    """
    Execute the predict function and return predictions.
    
    Args:
        predict_func: The predict function from a submission.
        train_df: Training DataFrame.
        test_df: Test DataFrame (without target).
        
    Returns:
        Array of predictions.
    """
    # Create copies to prevent submissions from modifying original data
    train_copy = train_df.copy()
    test_copy = test_df.copy()
    
    # Execute prediction
    predictions = predict_func(train_copy, test_copy)
    
    return predictions


def validate_predictions(
    predictions: Any,
    expected_length: int,
    submission_name: str
) -> np.ndarray:
    """
    Validate that predictions have the correct format and length.
    
    Args:
        predictions: Raw predictions from the submission.
        expected_length: Expected number of predictions.
        submission_name: Name of the submission (for error messages).
        
    Returns:
        Validated predictions as a numpy array.
        
    Raises:
        ValueError: If predictions are invalid.
    """
    # Handle None predictions
    if predictions is None:
        raise ValueError(f"predict() returned None")
    
    # Convert to numpy array if needed
    if isinstance(predictions, pd.Series):
        predictions = predictions.values
    elif isinstance(predictions, list):
        predictions = np.array(predictions)
    elif not isinstance(predictions, np.ndarray):
        raise ValueError(
            f"predict() returned unexpected type: {type(predictions).__name__}. "
            f"Expected: numpy array, pandas Series, or list"
        )
    
    # Flatten if needed (handle 2D arrays with single column)
    if predictions.ndim > 1:
        if predictions.shape[1] == 1:
            predictions = predictions.flatten()
        else:
            raise ValueError(
                f"predict() returned multi-column output with shape {predictions.shape}. "
                f"Expected 1D array or single-column 2D array"
            )
    
    # Validate length
    if len(predictions) != expected_length:
        raise ValueError(
            f"predict() returned {len(predictions)} predictions, "
            f"but expected {expected_length} (matching test set size)"
        )
    
    # Check for NaN values
    nan_count = np.sum(pd.isna(predictions))
    if nan_count > 0:
        raise ValueError(
            f"predict() returned {nan_count} NaN values out of {len(predictions)} predictions"
        )
    
    return predictions


# =============================================================================
# Metrics Computation
# =============================================================================

def determine_task_type(ground_truth: np.ndarray) -> bool:
    """
    Determine if the task is classification or regression.
    
    Args:
        ground_truth: Array of ground truth values.
        
    Returns:
        True if classification, False if regression.
    """
    unique_values = np.unique(ground_truth)
    
    # If few unique values and they're integers, likely classification
    if len(unique_values) <= 20:
        if np.all(ground_truth == ground_truth.astype(int)):
            return True
    
    return False


def compute_metrics(
    predictions: np.ndarray,
    ground_truth: np.ndarray
) -> EvaluationMetrics:
    """
    Compute all evaluation metrics for a submission.
    
    For classification tasks: Accuracy, Precision, Recall, F1, MSE, Confusion Matrix
    For regression tasks: MSE (classification metrics set to 0)
    
    Args:
        predictions: Predicted values.
        ground_truth: True values.
        
    Returns:
        EvaluationMetrics dataclass with all computed metrics.
    """
    is_classification = determine_task_type(ground_truth)
    
    # Always compute MSE
    mse = mean_squared_error(ground_truth, predictions)
    
    if is_classification:
        # Round predictions for classification metrics
        pred_classes = np.round(predictions).astype(int)
        true_classes = ground_truth.astype(int)
        
        # Get unique labels for multi-class handling
        labels = np.unique(np.concatenate([true_classes, pred_classes]))
        
        # Determine averaging strategy based on number of classes
        average = 'binary' if len(labels) == 2 else 'weighted'
        
        accuracy = accuracy_score(true_classes, pred_classes)
        precision = precision_score(
            true_classes, pred_classes, average=average, zero_division=0
        )
        recall = recall_score(
            true_classes, pred_classes, average=average, zero_division=0
        )
        f1 = f1_score(
            true_classes, pred_classes, average=average, zero_division=0
        )
        
        # Compute confusion matrix
        conf_mat = confusion_matrix(true_classes, pred_classes, labels=labels)
        
        return EvaluationMetrics(
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            mse=mse,
            is_classification=True,
            confusion_mat=conf_mat,
            class_labels=labels.tolist()
        )
    else:
        # For regression, classification metrics are not meaningful
        return EvaluationMetrics(
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            mse=mse,
            is_classification=False,
            confusion_mat=None,
            class_labels=None
        )


# =============================================================================
# Visualization Functions
# =============================================================================

def create_confusion_matrix_plot(
    confusion_mat: np.ndarray,
    labels: List,
    output_path: Path
) -> None:
    """
    Create and save a confusion matrix heatmap.
    
    Args:
        confusion_mat: The confusion matrix array.
        labels: Class labels for axes.
        output_path: Path to save the plot.
    """
    plt.figure(figsize=(8, 6))
    
    sns.heatmap(
        confusion_mat,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'}
    )
    
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_points_visualization(
    test_df: pd.DataFrame,
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    output_path_original: Path,
    output_path_predicted: Path,
    output_path_comparison: Path
) -> None:
    """
    Create 3D visualizations of original and predicted points.
    
    Args:
        test_df: Test DataFrame with x, y coordinates.
        predictions: Predicted z values.
        ground_truth: True z values.
        output_path_original: Path to save original points plot.
        output_path_predicted: Path to save predicted points plot.
        output_path_comparison: Path to save comparison plot.
    """
    from mpl_toolkits.mplot3d import Axes3D
    
    # Extract x, y coordinates
    x_coords = test_df['x'].values if 'x' in test_df.columns else test_df.iloc[:, 0].values
    y_coords = test_df['y'].values if 'y' in test_df.columns else test_df.iloc[:, 1].values
    
    # Determine color/z limits for consistent scaling
    vmin = min(ground_truth.min(), predictions.min())
    vmax = max(ground_truth.max(), predictions.max())
    
    # Plot 1: Original points (ground truth) - 3D
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    scatter1 = ax1.scatter(x_coords, y_coords, ground_truth, 
                           c=ground_truth, cmap='viridis', 
                           s=50, alpha=0.7, edgecolors='k', linewidths=0.3,
                           vmin=vmin, vmax=vmax)
    ax1.set_xlabel('x', fontsize=12)
    ax1.set_ylabel('y', fontsize=12)
    ax1.set_zlabel('z', fontsize=12)
    ax1.set_title('Original Points (Ground Truth)', fontsize=14, fontweight='bold')
    cbar1 = fig1.colorbar(scatter1, ax=ax1, shrink=0.6, pad=0.1)
    cbar1.set_label('z value', fontsize=10)
    ax1.view_init(elev=25, azim=45)
    plt.tight_layout()
    plt.savefig(output_path_original, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Predicted points - 3D
    fig2 = plt.figure(figsize=(10, 8))
    ax2 = fig2.add_subplot(111, projection='3d')
    scatter2 = ax2.scatter(x_coords, y_coords, predictions,
                           c=predictions, cmap='viridis',
                           s=50, alpha=0.7, edgecolors='k', linewidths=0.3,
                           vmin=vmin, vmax=vmax)
    ax2.set_xlabel('x', fontsize=12)
    ax2.set_ylabel('y', fontsize=12)
    ax2.set_zlabel('z', fontsize=12)
    ax2.set_title('Predicted Points (Submission Output)', fontsize=14, fontweight='bold')
    cbar2 = fig2.colorbar(scatter2, ax=ax2, shrink=0.6, pad=0.1)
    cbar2.set_label('z value', fontsize=10)
    ax2.view_init(elev=25, azim=45)
    plt.tight_layout()
    plt.savefig(output_path_predicted, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Side-by-side 3D comparison
    fig3 = plt.figure(figsize=(16, 7))
    
    ax3a = fig3.add_subplot(121, projection='3d')
    scatter3a = ax3a.scatter(x_coords, y_coords, ground_truth,
                              c=ground_truth, cmap='viridis',
                              s=40, alpha=0.7, edgecolors='k', linewidths=0.2,
                              vmin=vmin, vmax=vmax)
    ax3a.set_xlabel('x', fontsize=11)
    ax3a.set_ylabel('y', fontsize=11)
    ax3a.set_zlabel('z', fontsize=11)
    ax3a.set_title('Ground Truth', fontsize=12, fontweight='bold')
    ax3a.view_init(elev=25, azim=45)
    fig3.colorbar(scatter3a, ax=ax3a, shrink=0.5, pad=0.1, label='z')
    
    ax3b = fig3.add_subplot(122, projection='3d')
    scatter3b = ax3b.scatter(x_coords, y_coords, predictions,
                              c=predictions, cmap='viridis',
                              s=40, alpha=0.7, edgecolors='k', linewidths=0.2,
                              vmin=vmin, vmax=vmax)
    ax3b.set_xlabel('x', fontsize=11)
    ax3b.set_ylabel('y', fontsize=11)
    ax3b.set_zlabel('z', fontsize=11)
    ax3b.set_title('Predictions', fontsize=12, fontweight='bold')
    ax3b.view_init(elev=25, azim=45)
    fig3.colorbar(scatter3b, ax=ax3b, shrink=0.5, pad=0.1, label='z')
    
    plt.suptitle('Ground Truth vs Predictions (3D)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path_comparison, dpi=150, bbox_inches='tight')
    plt.close()


def create_error_plot(
    predictions: np.ndarray,
    ground_truth: np.ndarray,
    output_path: Path,
    is_classification: bool
) -> None:
    """
    Create and save an error visualization plot.
    
    For classification: Prediction distribution comparison
    For regression: Scatter plot of predicted vs actual values
    
    Args:
        predictions: Predicted values.
        ground_truth: True values.
        output_path: Path to save the plot.
        is_classification: Whether this is a classification task.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    if is_classification:
        # Plot 1: Prediction distribution using Counter (handles negative values)
        pred_classes = np.round(predictions).astype(int)
        true_classes = ground_truth.astype(int)
        
        # Get all unique labels and sort them
        all_labels = sorted(set(true_classes) | set(pred_classes))
        
        # Count occurrences using Counter (handles any integer values)
        true_counter = Counter(true_classes)
        pred_counter = Counter(pred_classes)
        
        # Build counts arrays aligned with labels
        true_counts = [true_counter.get(label, 0) for label in all_labels]
        pred_counts = [pred_counter.get(label, 0) for label in all_labels]
        
        ax1 = axes[0]
        x = np.arange(len(all_labels))
        width = 0.35
        
        ax1.bar(x - width/2, true_counts, width, label='True', color='steelblue', alpha=0.8)
        ax1.bar(x + width/2, pred_counts, width, label='Predicted', color='coral', alpha=0.8)
        ax1.set_xlabel('Class')
        ax1.set_ylabel('Count')
        ax1.set_title('Class Distribution: True vs Predicted')
        ax1.legend()
        ax1.set_xticks(x)
        ax1.set_xticklabels(all_labels)
        
        # Plot 2: Error indicators
        ax2 = axes[1]
        errors = (pred_classes != true_classes).astype(int)
        ax2.scatter(
            range(len(errors)),
            errors,
            c=['green' if e == 0 else 'red' for e in errors],
            alpha=0.5,
            s=20
        )
        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Error (0=Correct, 1=Incorrect)')
        ax2.set_title('Prediction Errors by Sample')
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['Correct', 'Incorrect'])
        
    else:
        # Regression plots
        # Plot 1: Scatter plot of predicted vs actual
        ax1 = axes[0]
        ax1.scatter(ground_truth, predictions, alpha=0.5, edgecolors='k', linewidths=0.5)
        
        # Add perfect prediction line
        min_val = min(ground_truth.min(), predictions.min())
        max_val = max(ground_truth.max(), predictions.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
        
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title('Predicted vs Actual Values')
        ax1.legend()
        
        # Plot 2: Residual plot
        ax2 = axes[1]
        residuals = predictions - ground_truth
        ax2.scatter(ground_truth, residuals, alpha=0.5, edgecolors='k', linewidths=0.5)
        ax2.axhline(y=0, color='r', linestyle='--')
        ax2.set_xlabel('Actual Values')
        ax2.set_ylabel('Residual (Predicted - Actual)')
        ax2.set_title('Residual Plot')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# PDF Report Generation
# =============================================================================

def generate_pdf_report(
    result: EvaluationResult,
    output_path: Path,
    temp_dir: Path
) -> None:
    """
    Generate a comprehensive PDF report for a submission.
    
    Args:
        result: EvaluationResult containing all evaluation data.
        output_path: Path to save the PDF report.
        temp_dir: Directory for temporary plot files.
    """
    # Initialize PDF document
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=letter,
        rightMargin=72,
        leftMargin=72,
        topMargin=72,
        bottomMargin=72
    )
    
    # Get styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        spaceAfter=30,
        textColor=colors.HexColor('#2c3e50')
    )
    
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading1'],
        fontSize=16,
        spaceAfter=12,
        spaceBefore=20,
        textColor=colors.HexColor('#34495e')
    )
    
    normal_style = styles['Normal']
    
    # Build document content
    story = []
    
    # Title
    story.append(Paragraph("ML Submission Evaluation Report", title_style))
    story.append(Spacer(1, 20))
    
    # Submission info table
    info_data = [
        ["Submission Name:", result.submission_name],
        ["Dataset:", result.dataset_name],
        ["Evaluation Date:", datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
        ["Status:", "✓ Success" if result.success else "✗ Failed"],
    ]
    
    if result.execution_time is not None:
        info_data.append(["Execution Time:", f"{result.execution_time:.3f} seconds"])
    
    info_table = Table(info_data, colWidths=[2*inch, 4*inch])
    info_table.setStyle(TableStyle([
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 11),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (0, -1), 'RIGHT'),
        ('ALIGN', (1, 0), (1, -1), 'LEFT'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
    ]))
    story.append(info_table)
    story.append(Spacer(1, 30))
    
    if not result.success:
        # Error report
        story.append(Paragraph("Error Details", heading_style))
        error_text = result.error_message or "Unknown error occurred"
        story.append(Paragraph(f"<font color='red'>{error_text}</font>", normal_style))
        
    else:
        # Success report with metrics
        metrics = result.metrics
        
        # Task type
        task_type = "Classification" if metrics.is_classification else "Regression"
        story.append(Paragraph(f"Task Type: {task_type}", heading_style))
        story.append(Spacer(1, 10))
        
        # Metrics section
        story.append(Paragraph("Evaluation Metrics", heading_style))
        
        if metrics.is_classification:
            metrics_data = [
                ["Metric", "Value"],
                ["Accuracy", f"{metrics.accuracy:.6e}"],
                ["Precision", f"{metrics.precision:.6e}"],
                ["Recall", f"{metrics.recall:.6e}"],
                ["F1 Score", f"{metrics.f1:.6e}"],
                ["MSE", f"{metrics.mse:.6e}"],
            ]
        else:
            metrics_data = [
                ["Metric", "Value"],
                ["Mean Squared Error (MSE)", f"{metrics.mse:.6e}"],
                ["Root MSE (RMSE)", f"{np.sqrt(metrics.mse):.6e}"],
            ]
        
        metrics_table = Table(metrics_data, colWidths=[3*inch, 2*inch])
        metrics_table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 11),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#3498db')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 1, colors.HexColor('#bdc3c7')),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#ecf0f1')]),
        ]))
        story.append(metrics_table)
        story.append(Spacer(1, 20))
        
        # Visualizations
        if metrics.is_classification and metrics.confusion_mat is not None:
            # Confusion Matrix
            story.append(Paragraph("Confusion Matrix", heading_style))
            
            cm_plot_path = temp_dir / f"cm_{result.submission_name}.png"
            create_confusion_matrix_plot(
                metrics.confusion_mat,
                metrics.class_labels,
                cm_plot_path
            )
            
            if cm_plot_path.exists():
                story.append(Image(str(cm_plot_path), width=5*inch, height=4*inch))
            
            story.append(Spacer(1, 20))
        
        # Error/Scatter plot
        story.append(Paragraph(
            "Prediction Analysis" if metrics.is_classification else "Regression Analysis",
            heading_style
        ))
        
        error_plot_path = temp_dir / f"error_{result.submission_name}.png"
        create_error_plot(
            result.predictions,
            result.ground_truth,
            error_plot_path,
            metrics.is_classification
        )
        
        if error_plot_path.exists():
            story.append(Image(str(error_plot_path), width=6*inch, height=2.5*inch))
        
        # Points visualization (if x, y coordinates available)
        if result.test_df is not None and 'x' in result.test_df.columns and 'y' in result.test_df.columns:
            # Page break for visualizations
            story.append(PageBreak())
            
            # Original points visualization
            story.append(Paragraph("Original Points (Ground Truth)", heading_style))
            original_plot_path = temp_dir / f"original_{result.submission_name}.png"
            predicted_plot_path = temp_dir / f"predicted_{result.submission_name}.png"
            comparison_plot_path = temp_dir / f"comparison_{result.submission_name}.png"
            
            create_points_visualization(
                result.test_df,
                result.predictions,
                result.ground_truth,
                original_plot_path,
                predicted_plot_path,
                comparison_plot_path
            )
            
            if original_plot_path.exists():
                story.append(Image(str(original_plot_path), width=5.5*inch, height=4*inch))
            
            story.append(Spacer(1, 20))
            
            # Predicted points visualization
            story.append(Paragraph("Predicted Points (Submission Output)", heading_style))
            
            if predicted_plot_path.exists():
                story.append(Image(str(predicted_plot_path), width=5.5*inch, height=4*inch))
            
            story.append(PageBreak())
            
            # Side-by-side comparison
            story.append(Paragraph("Comparison: Ground Truth vs Predictions", heading_style))
            
            if comparison_plot_path.exists():
                story.append(Image(str(comparison_plot_path), width=6.5*inch, height=2.8*inch))
    
    # Build the PDF
    doc.build(story)
    print(f"    📄 Report saved: {output_path.name}")


# =============================================================================
# Main Evaluation Pipeline
# =============================================================================

def evaluate_submission(
    submission_path: Path,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    ground_truth: np.ndarray,
    dataset_name: str = "default"
) -> EvaluationResult:
    """
    Evaluate a single submission against a dataset.
    
    Args:
        submission_path: Path to the submission .py file.
        train_df: Training DataFrame.
        test_df: Test DataFrame.
        ground_truth: Ground truth values for evaluation.
        dataset_name: Name of the dataset being evaluated.
        
    Returns:
        EvaluationResult with metrics or error information.
    """
    submission_name = submission_path.stem
    
    print(f"\n  📝 Evaluating: {submission_name} on dataset '{dataset_name}'")
    
    try:
        # Step 1: Load the predict function
        import time
        start_time = time.time()
        
        predict_func = load_predict_function(submission_path)
        print(f"    ✓ Loaded predict() function")
        
        # Step 2: Execute prediction
        predictions_raw = execute_prediction(predict_func, train_df, test_df)
        print(f"    ✓ Executed predictions")
        
        # Step 3: Validate predictions
        predictions = validate_predictions(
            predictions_raw,
            expected_length=len(ground_truth),
            submission_name=submission_name
        )
        print(f"    ✓ Validated predictions (n={len(predictions)})")
        
        execution_time = time.time() - start_time
        
        # Step 4: Compute metrics
        metrics = compute_metrics(predictions, ground_truth)
        task_type = "Classification" if metrics.is_classification else "Regression"
        print(f"    ✓ Computed metrics ({task_type})")
        
        if metrics.is_classification:
            print(f"      Accuracy: {metrics.accuracy:.4f}, F1: {metrics.f1:.4f}")
        else:
            print(f"      MSE: {metrics.mse:.6f}, RMSE: {np.sqrt(metrics.mse):.6f}")
        
        return EvaluationResult(
            submission_name=submission_name,
            dataset_name=dataset_name,
            success=True,
            metrics=metrics,
            predictions=predictions,
            ground_truth=ground_truth,
            test_df=test_df,
            execution_time=execution_time
        )
        
    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        print(f"    ✗ Error: {error_msg}")
        
        return EvaluationResult(
            submission_name=submission_name,
            dataset_name=dataset_name,
            success=False,
            error_message=error_msg
        )


def run_evaluation_pipeline(folder_path: Path) -> None:
    """
    Run the complete evaluation pipeline for all submissions against all datasets.
    
    Supports two modes:
    1. Legacy mode: CSV files directly in submissions folder
    2. Multi-dataset mode: Subfolders containing CSV files
    
    Args:
        folder_path: Path to the submissions folder.
    """
    folder = folder_path.resolve()
    
    print("=" * 70)
    print("🎯 Evaluatinator - ML Submission Evaluator")
    print("=" * 70)
    print(f"\n📁 Submissions folder: {folder}")
    
    # Validate folder exists
    if not folder.exists():
        print(f"\n❌ Error: Folder not found: {folder}")
        print(f"\n💡 Tip: Make sure to place your files in the ./submissions folder:")
        print(f"    - train.csv, test.csv, test_internal.csv (in root or subfolders)")
        print(f"    - your_submission.py")
        sys.exit(1)
    
    # Discover datasets
    print("\n📂 Discovering datasets...")
    datasets = discover_datasets(folder)
    
    if not datasets:
        print(f"\n❌ No valid datasets found!")
        print(f"\n💡 Expected structure (Option 1 - Single dataset):")
        print(f"    ./submissions/")
        print(f"    ├── train.csv")
        print(f"    ├── test.csv")
        print(f"    ├── test_internal.csv")
        print(f"    └── submission.py")
        print(f"\n💡 Expected structure (Option 2 - Multiple datasets):")
        print(f"    ./submissions/")
        print(f"    ├── dataset1/")
        print(f"    │   ├── train.csv")
        print(f"    │   ├── test.csv")
        print(f"    │   └── test_internal.csv")
        print(f"    ├── dataset2/")
        print(f"    │   ├── train.csv")
        print(f"    │   ├── test.csv")
        print(f"    │   └── test_internal.csv")
        print(f"    └── submission.py")
        sys.exit(1)
    
    print(f"\n✓ Found {len(datasets)} dataset(s): {', '.join(d.name for d in datasets)}")
    
    # Discover submissions
    submissions = discover_submissions(folder)
    
    if not submissions:
        print(f"\n⚠️  No .py submission files found in {folder}")
        print(f"\n💡 Create a Python file with a predict(train_df, test_df) function.")
        print(f"   See sample_submission.py for an example.")
        sys.exit(0)
    
    print(f"\n📋 Found {len(submissions)} submission(s):")
    for sub in submissions:
        print(f"    - {sub.name}")
    
    # Create output directories
    reports_dir = folder / REPORTS_SUBDIR
    reports_dir.mkdir(exist_ok=True)
    
    temp_dir = reports_dir / "temp_plots"
    temp_dir.mkdir(exist_ok=True)
    
    # Evaluate each submission against each dataset
    print("\n" + "=" * 70)
    print("Starting Evaluation")
    print("=" * 70)
    
    results: List[EvaluationResult] = []
    
    for dataset in datasets:
        print(f"\n{'─' * 50}")
        print(f"📊 Dataset: {dataset.name}")
        print(f"{'─' * 50}")
        
        for submission_path in submissions:
            result = evaluate_submission(
                submission_path,
                dataset.train_df,
                dataset.test_df,
                dataset.ground_truth,
                dataset_name=dataset.name
            )
            results.append(result)
            
            # Generate PDF report with dataset name
            if len(datasets) > 1:
                report_name = f"report_{result.submission_name}_{dataset.name}.pdf"
            else:
                report_name = f"report_{result.submission_name}.pdf"
            
            report_path = reports_dir / report_name
            generate_pdf_report(result, report_path, temp_dir)
    
    # Clean up temporary plot files
    for temp_file in temp_dir.glob("*.png"):
        temp_file.unlink()
    temp_dir.rmdir()
    
    # Print summary
    print("\n" + "=" * 70)
    print("Evaluation Summary")
    print("=" * 70)
    
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    
    print(f"\n✓ Successful: {len(successful)}/{len(results)}")
    print(f"✗ Failed: {len(failed)}/{len(results)}")
    
    if successful:
        # Group results by dataset
        for dataset in datasets:
            dataset_results = [r for r in successful if r.dataset_name == dataset.name]
            
            if not dataset_results:
                continue
            
            print(f"\n📊 Results for dataset '{dataset.name}':")
            
            # Determine if classification or regression
            is_classification = dataset_results[0].metrics.is_classification
            
            if is_classification:
                print(f"\n{'Submission':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'MSE':>12}")
                print("-" * 77)
                
                for r in dataset_results:
                    m = r.metrics
                    print(f"{r.submission_name:<25} {m.accuracy:>10.6e} {m.precision:>10.6e} {m.recall:>10.6e} {m.f1:>10.6e} {m.mse:>12.6e}")
            else:
                print(f"\n{'Submission':<35} {'MSE':>15} {'RMSE':>15}")
                print("-" * 65)
                
                for r in dataset_results:
                    m = r.metrics
                    print(f"{r.submission_name:<35} {m.mse:>15.6e} {np.sqrt(m.mse):>15.6e}")
    
    if failed:
        print("\n❌ Failed Evaluations:")
        for r in failed:
            print(f"  - {r.submission_name} on {r.dataset_name}: {r.error_message}")
    
    print(f"\n📁 Reports saved to: {reports_dir}")
    print("\n✅ Evaluation complete!")


# =============================================================================
# Command Line Interface
# =============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate machine learning submissions and generate PDF reports.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  python evaluator.py                      # Use default ./submissions folder
  python evaluator.py --folder ./my_data   # Use custom folder

Expected folder structure (Option 1 - Single dataset):
  ./submissions/
  ├── train.csv
  ├── test.csv  
  ├── test_internal.csv
  ├── submission_1.py
  └── submission_2.py

Expected folder structure (Option 2 - Multiple datasets):
  ./submissions/
  ├── dataset1/
  │   ├── train.csv
  │   ├── test.csv
  │   └── test_internal.csv
  ├── dataset2/
  │   ├── train.csv
  │   ├── test.csv
  │   └── test_internal.csv
  ├── submission_1.py
  └── submission_2.py
        """
    )
    
    parser.add_argument(
        '-f', '--folder',
        type=Path,
        default=DEFAULT_SUBMISSIONS_FOLDER,
        help=f"Path to the submissions folder (default: ./submissions)"
    )
    
    return parser.parse_args()


# =============================================================================
# Entry Point
# =============================================================================

if __name__ == "__main__":
    args = parse_arguments()
    run_evaluation_pipeline(args.folder)
