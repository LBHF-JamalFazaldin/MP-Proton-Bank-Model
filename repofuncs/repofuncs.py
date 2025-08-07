import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import os
import sys

current_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(current_file))
sys.path.insert(0, project_root)

from myfuncs import myfuncs as mf
 

def plot_all_distributions(df):
    """
    Plots the distribution of all columns in a DataFrame.
    - Numeric columns → Histogram
    - Categorical columns → Bar plot of value counts
    """
    for col in df.columns:
        plt.figure(figsize=(8, 4))
        if pd.api.types.is_numeric_dtype(df[col]):
            # Plot numeric column distribution
            sns.histplot(df[col].dropna(), kde=True, bins=30)
            plt.title(f'Distribution of {col}')
        else:
            # Plot categorical column value counts
            df[col].value_counts(normalize=True).plot(kind='bar')
            plt.title(f'Value Counts of {col}')
            plt.ylabel('Proportion')
        plt.xlabel(col)
        plt.tight_layout()
        plt.show()

def plot_correlation_matrix(df):
    corr = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.show()

from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant
 
def calculate_vif(df):
    """
    Calculates Variance Inflation Factor (VIF) for each feature in a DataFrame.
    Only for numeric features.
    """
    X = add_constant(df.select_dtypes(include='number').dropna())
    vif_data = pd.DataFrame()
    vif_data['Feature'] = X.columns
    vif_data['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
    mf.display(vif_data)
    return

def plot_confusion_matrix(cm, normalize='total'):
    """
    Plot a confusion matrix as percentages, with normalization options.
 
    Args:
        cm (np.ndarray): A 2x2 confusion matrix.
        normalize (str): 
            - 'total': Normalize each cell by the grand total (default).
            - True: Normalize each row by the actual class (row-wise).
    """
    if normalize == 'total':
        cm_percent = cm / cm.sum() * 100
        title = "Confusion Matrix (% of Total)"
    elif normalize:
        cm_percent = cm / cm.sum(axis=1, keepdims=True) * 100
        title = "Confusion Matrix (% of Actuals / True Labels)"
    else:
        raise ValueError("normalize must be 'total' or 'true'")
 
    fig, ax = plt.subplots(figsize=(8, 8))
    cax = ax.imshow(cm_percent, cmap='cool')
    plt.colorbar(cax, fraction=0.046, pad=0.04)
 
    # Axis labels
    ax.grid(False)
    ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
    ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    ax.set_ylim(1.5, -0.5)
 
    # Annotate each cell
    for i in range(2):
        for j in range(2):
            percent = cm_percent[i, j]
            ax.text(j, i, f"{percent:.1f}%", ha='center', va='center', color='black', fontsize=14)
 
    plt.title(title, fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_sigmoid_function(logits, **kwargs):
    sigmoid = lambda z: 1 / (1 + np.exp(-z))
    probabilities = sigmoid(logits)
    
    predictions = kwargs.get('predictions')
    y_test = kwargs.get('y_test')

    # Compare to true labels
    correct = predictions == y_test
    
    # Convert to numpy arrays if needed
    logits = np.array(logits)
    probabilities = np.array(probabilities)
    y_test = np.array(y_test)
    
    # Create figure
    plt.figure(figsize=(10, 6))
    
    # Plot predicted probabilities as gray squares
    plt.scatter(
        logits,
        probabilities,
        marker='s',
        color='gray',
        label='Predicted probability, $p(x_i)$',
        zorder=1
    )
    
    # Add dashed vertical lines from predicted to actual label
    for i in range(len(logits)):
        plt.plot(
            [logits[i], logits[i]],
            [y_test[i], probabilities[i]],
            color='gray',
            linestyle='dashed',
            linewidth=1,
            zorder=0,
            alpha=0.2
        )
    
    # Plot correct predictions as green circles
    plt.scatter(
        logits[correct],
        y_test[correct],
        color='green',
        marker='o',
        label='Correct predictions (actual responses), $y_i$',
        zorder=2
    )
    
    # Plot incorrect predictions as red Xs
    plt.scatter(
        logits[~correct],
        y_test[~correct],
        color='red',
        marker='x',
        s=100,
        label='Incorrect predictions',
        zorder=3
    )
    
    # Logistic regression curve
    logit_range = np.linspace(min(logits), max(logits), 300)
    plt.plot(
        logit_range,
        sigmoid(logit_range),
        color='black',
        linestyle='-',
        label='Estimated logistic regression line, $p(x)$',
        zorder=0
    )
    
    # Threshold line
    plt.axhline(0.5, linestyle='--', color='gray')
    
    # Formatting
    plt.xlabel('Logit (Linear Score $f(x)$)')
    plt.ylabel('Predicted Probability $p(x)$')
    plt.title('Logistic Regression: Prediction Accuracy and Probability')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def evaluate_coeffcients(model, X):
    # Create DataFrame
    effects = model.coef_[0]
    factors = X.columns
    
    effect_df = pd.DataFrame({'Effect': effects}, index=factors)
    effect_df.sort_values(by='Effect', inplace=True)
    
    # Color-code: green for positive, red for negative
    colors = ['green' if val > 0 else 'red' for val in effect_df['Effect']]
    
    # Plotting
    fig, ax = plt.subplots(figsize=(12, 12))
    bars = ax.barh(effect_df.index, effect_df['Effect'], color=colors)
    
    # Add value labels to each bar
    for bar in bars:
        width = bar.get_width()
        label_x_pos = width + 0.01 if width > 0 else width - 0.01
        ax.text(label_x_pos, bar.get_y() + bar.get_height()/2,
                f'{width:.2f}', va='center', ha='left' if width > 0 else 'right')
    
    # Enhancements
    ax.set_title('Feature Effects from Model Coefficients', fontsize=16)
    ax.set_xlabel('Effect Size', fontsize=14)
    ax.set_ylabel('Feature', fontsize=14)
    ax.axvline(0, color='black', linewidth=0.8)  # vertical line at 0
    ax.grid(True, axis='x', linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.show()