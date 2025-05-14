import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, average_precision_score
)
import dagshub
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer

# Initialize MLflow tracking
try:
    # Initialize Dagshub
    dagshub.init(repo_owner='konstantine25b', repo_name='IEEE-CIS-Fraud-Detection', mlflow=True)
    print("DagsHub initialized successfully.")
    mlflow.set_experiment("IEEE-CIS-Fraud-Detection_logistic_regression")
    print(f"MLflow experiment set to: {mlflow.get_experiment_by_name('IEEE-CIS-Fraud-Detection_logistic_regression').name}")
    mlflow_active = True
except Exception as e:
    print(f"Could not initialize DagsHub or set MLflow experiment: {e}")
    print("Proceeding without MLflow tracking.")
    mlflow_active = False

# Start MLflow run
run_name = f"logistic_regression_{time.strftime('%Y%m%d_%H%M%S')}"
if mlflow_active:
    mlflow.start_run(run_name=run_name)
    print(f"MLflow run started with name: {run_name}")

# Load the original data
print("\n--- Loading Original Data from Kaggle ---")
try:
    identity_df = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')
    transaction_df = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
    print(f"Loaded identity data shape: {identity_df.shape}")
    print(f"Loaded transaction data shape: {transaction_df.shape}")
except FileNotFoundError:
    print("Error: One or both of the CSV files were not found. Please make sure the file paths are correct.")
    if mlflow_active:
        mlflow.end_run()
    exit()

# Load the preprocessing pipeline from MLflow
print("\n--- Loading Preprocessing Pipeline from MLflow ---")
# Replace with your actual run ID from the preprocessing pipeline
preprocessing_run_id = '962cdbe1451f4abe864ff349e123e7de'  # Example run ID
try:
    # Load the transaction pipeline model
    transaction_pipeline = mlflow.sklearn.load_model(f'runs:/{preprocessing_run_id}/transaction_pipeline_model')
    print("Loaded transaction preprocessing pipeline from MLflow.")
    
    # Load the identity pipeline model if it exists
    try:
        identity_pipeline = mlflow.sklearn.load_model(f'runs:/{preprocessing_run_id}/identity_pipeline_model')
        print("Loaded identity preprocessing pipeline from MLflow.")
        identity_pipeline_exists = True
    except:
        print("Identity preprocessing pipeline not found. Will only use transaction pipeline.")
        identity_pipeline_exists = False
    
    # Load the feature selection information
    try:
        selected_features = mlflow.artifacts.load_text(f'runs:/{preprocessing_run_id}/selected_features.txt').strip().split('\n')
        print(f"Loaded {len(selected_features)} selected features from MLflow.")
    except:
        print("Selected features list not found. Will use all features after preprocessing.")
        selected_features = None
except Exception as e:
    print(f"Error loading preprocessing pipeline from MLflow: {e}")
    print("Please ensure you have the correct run ID and the pipeline is properly saved.")
    if mlflow_active:
        mlflow.end_run()
    exit()

# Split the data
X_transaction = transaction_df.drop('isFraud', axis=1)
y_transaction = transaction_df['isFraud']

X_train, X_test, y_train, y_test = train_test_split(
    X_transaction, y_transaction, test_size=0.2, random_state=42, stratify=y_transaction
)
print(f"Train set shape: {X_train.shape}")
print(f"Test set shape: {X_test.shape}")

# Get the original feature names before preprocessing
original_feature_names = X_train.columns.tolist()
print(f"Original feature count: {len(original_feature_names)}")

# Apply the transaction preprocessing pipeline
print("Applying transaction preprocessing pipeline...")
X_train_processed = transaction_pipeline.transform(X_train)
X_test_processed = transaction_pipeline.transform(X_test)

# Convert to DataFrame with feature names that match the preprocessing pipeline
# First, use generic names
feature_names = [f"feature_{i}" for i in range(X_train_processed.shape[1])]
X_train_final = pd.DataFrame(X_train_processed, columns=feature_names)
X_test_final = pd.DataFrame(X_test_processed, columns=feature_names)

# Apply feature selection if available
if selected_features is not None:
    print(f"Applying feature selection to keep {len(selected_features)} features...")
    
    # Print some of the selected feature names to understand their format
    print(f"Sample selected feature names: {selected_features[:5]}")
    
    # Check if any of the selected features match the original feature names
    original_matches = [f for f in selected_features if f in original_feature_names]
    if original_matches:
        print(f"Found {len(original_matches)} features that match original feature names.")
        
        # Create a mapping from original feature names to processed feature indices
        # This is a simplified approach - in reality, the mapping might be more complex
        feature_mapping = {}
        for i, feature in enumerate(original_feature_names):
            if i < len(feature_names):
                feature_mapping[feature] = feature_names[i]
        
        # Map selected features to their corresponding processed features
        mapped_features = []
        for feature in selected_features:
            if feature in feature_mapping:
                mapped_features.append(feature_mapping[feature])
        
        if mapped_features:
            print(f"Mapped {len(mapped_features)} selected features to processed features.")
            
            # Apply the selection
            X_train_final = X_train_final[mapped_features]
            X_test_final = X_test_final[mapped_features]
        else:
            print("Could not map any selected features to processed features.")
            print("Using all processed features.")
    else:
        # If selected features don't match original names, they might be indices or have a different format
        print("Selected features don't match original feature names.")
        
        # Print all selected features to understand their format
        print(f"First 5 selected features: {selected_features[:5]}")
        print(f"Last 5 selected features: {selected_features[-5:]}")
        
        # Check what columns are actually available in the processed data
        print(f"Available columns in processed data: {X_train_final.columns[:10]}...")
        
        # Extract the indices from the selected feature names
        feature_indices = []
        for feature in selected_features:
            # Try different patterns to extract indices
            if '_x' in feature or '_y' in feature:
                # Format like '0_x' or '1_y'
                parts = feature.split('_')
                if parts[0].isdigit():
                    feature_indices.append(int(parts[0]))
            elif feature.isdigit():
                # Format like '68'
                feature_indices.append(int(feature))
        
        # Get the available columns in the processed data
        available_columns = X_train_final.columns.tolist()
        
        # Create a new DataFrame with the original feature names
        X_train_renamed = pd.DataFrame()
        X_test_renamed = pd.DataFrame()
        
        # Map each selected feature to a column in the processed data
        for i, feature in enumerate(selected_features):
            if i < len(available_columns):
                # Use the i-th column from the processed data for the i-th selected feature
                # Extract as a Series using .iloc to avoid the DataFrame issue
                X_train_renamed[feature] = X_train_final.iloc[:, i].values
                X_test_renamed[feature] = X_test_final.iloc[:, i].values
            else:
                print(f"Warning: Not enough columns in processed data for feature {feature}")
        
        # Use the renamed DataFrames
        X_train_final = X_train_renamed
        X_test_final = X_test_renamed
        
        print(f"Final train set shape after preprocessing: {X_train_final.shape}")
        print(f"Final test set shape after preprocessing: {X_test_final.shape}")
        print(f"Feature names in final dataset: {X_train_final.columns.tolist()[:5]}...")
else:
    print("No selected features provided. Using all features from the preprocessing pipeline.")

print(f"Final train set shape after preprocessing: {X_train_final.shape}")
print(f"Final test set shape after preprocessing: {X_test_final.shape}")

# Check for NaN values in the processed data
print("\n--- Checking for NaN Values in Processed Data ---")
train_nan_count = X_train_final.isna().sum().sum()
test_nan_count = X_test_final.isna().sum().sum()

print(f"Number of NaN values in training data: {train_nan_count}")
print(f"Number of NaN values in testing data: {test_nan_count}")

if train_nan_count > 0 or test_nan_count > 0:
    print("WARNING: NaN values found in processed data!")
    
    # Get columns with NaN values
    train_nan_cols = X_train_final.columns[X_train_final.isna().any()].tolist()
    test_nan_cols = X_test_final.columns[X_test_final.isna().any()].tolist()
    
    if train_nan_cols:
        print(f"Training data columns with NaN values: {train_nan_cols}")
        print(f"NaN counts per column: \n{X_train_final[train_nan_cols].isna().sum()}")
    
    if test_nan_cols:
        print(f"Testing data columns with NaN values: {test_nan_cols}")
        print(f"NaN counts per column: \n{X_test_final[test_nan_cols].isna().sum()}")
    
    # Fill NaN values with median as a quick fix
    print("Filling NaN values with column medians...")
    for col in train_nan_cols:
        median_val = X_train_final[col].median()
        X_train_final[col] = X_train_final[col].fillna(median_val)
    
    for col in test_nan_cols:
        if col in X_train_final.columns:
            median_val = X_train_final[col].median()
        else:
            median_val = X_test_final[col].median()
        X_test_final[col] = X_test_final[col].fillna(median_val)
    
    # Verify NaN values are gone
    train_nan_count_after = X_train_final.isna().sum().sum()
    test_nan_count_after = X_test_final.isna().sum().sum()
    print(f"NaN values after filling - training data: {train_nan_count_after}")
    print(f"NaN values after filling - testing data: {test_nan_count_after}")
else:
    print("No NaN values found in processed data. Preprocessing pipeline handled missing values correctly.")

# Log preprocessing information
if mlflow_active:
    mlflow.log_param("original_features", X_train.shape[1])
    mlflow.log_param("final_features", X_train_final.shape[1])

# After loading and preprocessing the data, before training the model:

print("\n--- Setting up Cross-Validation and Hyperparameter Tuning ---")

# Create a pipeline with scaling and logistic regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42, n_jobs=-1))
])

# Define a more focused hyperparameter grid
param_grid = {
    'classifier__C': [0.01, 1.0, 10.0],  # Reduced regularization options
    'classifier__penalty': ['l2'],  # Focus on L2 regularization which is more stable
    'classifier__solver': ['liblinear'],  # Just one solver
    'classifier__class_weight': ['balanced', {0: 1, 1: 50}]  # Fewer class weight options
}

# Set up k-fold cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define custom scoring metrics for imbalanced data
scoring = {
    'average_precision': make_scorer(average_precision_score),
    'roc_auc': make_scorer(roc_auc_score),
    'recall': make_scorer(recall_score),
    'precision': make_scorer(precision_score)
}

# Set up grid search with cross-validation
grid_search = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=cv, 
    scoring=scoring,
    refit='average_precision',  # Optimize for average precision
    verbose=1,
    n_jobs=-1
)

# Train the model with grid search
print("Training model with cross-validation and hyperparameter tuning...")
start_time = time.time()
grid_search.fit(X_train_final, y_train)
training_time = time.time() - start_time

print(f"Model trained in {training_time:.2f} seconds")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation score (Average Precision): {grid_search.best_score_:.4f}")

# Use the best model for predictions
model = grid_search.best_estimator_

# Log best parameters and cross-validation scores
if mlflow_active:
    mlflow.log_params(grid_search.best_params_)
    mlflow.log_metric("best_cv_score", grid_search.best_score_)
    mlflow.log_metric("training_time", training_time)
    
    # Log all CV results
    cv_results = pd.DataFrame(grid_search.cv_results_)
    mlflow.log_text(cv_results.to_string(), "cv_results.txt")

# Evaluate the model
print("\n--- Evaluating Model Performance ---")

# Make predictions
y_pred = model.predict(X_test_final)
y_pred_proba = model.predict_proba(X_test_final)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)
avg_precision = average_precision_score(y_test, y_pred_proba)

# Print metrics
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"Average Precision: {avg_precision:.4f}")

# Log metrics
if mlflow_active:
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1", f1)
    mlflow.log_metric("roc_auc", roc_auc)
    mlflow.log_metric("avg_precision", avg_precision)

# Print classification report
print("\nClassification Report:")
report = classification_report(y_test, y_pred)
print(report)

# Log classification report
if mlflow_active:
    mlflow.log_text(report, "classification_report.txt")

# Create and log confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
cm_path = "confusion_matrix.png"
plt.savefig(cm_path)
plt.close()

# Create and log ROC curve
plt.figure(figsize=(8, 6))
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.tight_layout()
roc_path = "roc_curve.png"
plt.savefig(roc_path)
plt.close()

# Create and log Precision-Recall curve
plt.figure(figsize=(8, 6))
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
plt.plot(recall_curve, precision_curve, label=f'PR Curve (AP = {avg_precision:.4f})')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.tight_layout()
pr_path = "precision_recall_curve.png"
plt.savefig(pr_path)
plt.close()

# Log artifacts
if mlflow_active:
    mlflow.log_artifact(cm_path)
    mlflow.log_artifact(roc_path)
    mlflow.log_artifact(pr_path)
    
    # Log the model
    mlflow.sklearn.log_model(model, "logistic_regression_model")
    
    # Log feature coefficients
    coef_df = pd.DataFrame({
        'Feature': X_train_final.columns,
        'Coefficient': model.coef_[0],
        'Abs_Coefficient': np.abs(model.coef_[0])
    }).sort_values('Abs_Coefficient', ascending=False)
    
    mlflow.log_text(coef_df.to_string(), "feature_coefficients.txt")
    
    # Create and log feature importance plot
    plt.figure(figsize=(12, 10))
    top_n = 20
    top_coef = coef_df.head(top_n)
    colors = ['red' if c < 0 else 'green' for c in top_coef['Coefficient']]
    plt.barh(top_coef['Feature'], top_coef['Abs_Coefficient'], color=colors)
    plt.title(f'Top {top_n} Feature Coefficients (Red = Negative, Green = Positive)')
    plt.xlabel('Absolute Coefficient Value')
    plt.tight_layout()
    coef_path = "feature_coefficients.png"
    plt.savefig(coef_path)
    plt.close()
    
    mlflow.log_artifact(coef_path)

# Log NaN check results
if mlflow_active:
    mlflow.log_param("train_nan_count_before", train_nan_count)
    mlflow.log_param("test_nan_count_before", test_nan_count)
    if train_nan_count > 0 or test_nan_count > 0:
        mlflow.log_param("train_nan_count_after", train_nan_count_after)
        mlflow.log_param("test_nan_count_after", test_nan_count_after)

# End MLflow run
if mlflow_active:
    mlflow.end_run()
    print("MLflow run completed and artifacts logged.")

print("\n--- Logistic Regression Training and Evaluation Completed ---") 