import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import time
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, average_precision_score
)
import dagshub
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import make_scorer

# Initialize MLflow tracking
try:
    # Initialize Dagshub
    dagshub.init(repo_owner='konstantine25b', repo_name='IEEE-CIS-Fraud-Detection', mlflow=True)
    print("DagsHub initialized successfully.")
    mlflow.set_experiment("IEEE-CIS-Fraud-Detection_xgboost")
    print(f"MLflow experiment set to: {mlflow.get_experiment_by_name('IEEE-CIS-Fraud-Detection_xgboost').name}")
    mlflow_active = True
except Exception as e:
    print(f"Could not initialize DagsHub or set MLflow experiment: {e}")
    print("Proceeding without MLflow tracking.")
    mlflow_active = False

# Start MLflow run
run_name = f"xgboost_{time.strftime('%Y%m%d_%H%M%S')}"
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

# Define the pipeline with scaler and XGBoost classifier
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        use_label_encoder=False,
        eval_metric='auc'
    ))
])

# Define hyperparameter grid focused on improving precision for minority class
param_grid = {
    'classifier__n_estimators': [200],  # Reduced options
    'classifier__max_depth': [4, 6],  # Reduced options
    'classifier__learning_rate': [0.01],  # Only one learning rate
    'classifier__scale_pos_weight': [25, 35],  # Reduced options
    'classifier__min_child_weight': [3],  # Only one option
    'classifier__subsample': [0.8],  # Only one option
    'classifier__colsample_bytree': [0.8],  # Only one option
    'classifier__gamma': [0.1],  # Only one option
    'classifier__reg_alpha': [0.1],  # Only one option
    'classifier__reg_lambda': [1.0]  # Only one option
}

# This reduces combinations from 2304 to just 4 (2×1×2×1×1×1×1×1×1)

# Reduce cross-validation folds
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Sample 5% of the training data for hyperparameter tuning (reduced from 10%)
sample_size = int(0.05 * len(X_train_final))
indices = np.random.choice(len(X_train_final), sample_size, replace=False)
X_train_sample = X_train_final.iloc[indices]
y_train_sample = y_train.iloc[indices]

# Use the sampled data for grid search
print(f"Using {len(X_train_sample)} samples for hyperparameter tuning...")
grid_search = GridSearchCV(
    pipeline, 
    param_grid, 
    cv=cv, 
    scoring='f1',  # Optimize for F1 score which balances precision and recall
    verbose=1,
    n_jobs=-1
)
grid_search.fit(X_train_sample, y_train_sample)

# After finding best parameters, train final model on full dataset
print("Training final model with best parameters on full dataset...")
best_params = grid_search.best_params_

# Create a new XGBoost classifier with the best parameters
best_xgb = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=best_params['classifier__n_estimators'],
    max_depth=best_params['classifier__max_depth'],
    learning_rate=best_params['classifier__learning_rate'],
    scale_pos_weight=best_params['classifier__scale_pos_weight'],
    min_child_weight=best_params.get('classifier__min_child_weight', 1),
    subsample=best_params.get('classifier__subsample', 0.8),
    colsample_bytree=best_params.get('classifier__colsample_bytree', 0.8),
    gamma=best_params.get('classifier__gamma', 0),
    reg_alpha=best_params.get('classifier__reg_alpha', 0),
    reg_lambda=best_params.get('classifier__reg_lambda', 1),
    random_state=42,
    use_label_encoder=False,
    eval_metric='auc',
    n_jobs=-1
)

# Use a validation set for early stopping in the final model
X_train_fit, X_val, y_train_fit, y_val = train_test_split(
    X_train_final, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Apply StandardScaler to the data
scaler = StandardScaler()
X_train_fit_scaled = scaler.fit_transform(X_train_fit)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test_final)

# Train the model with more early stopping rounds
best_xgb.fit(
    X_train_fit_scaled, y_train_fit,
    eval_set=[(X_val_scaled, y_val)],
    verbose=True,  # Show progress
    early_stopping_rounds=100  # More patience for early stopping
)

# Find optimal threshold for better precision-recall balance
print("\n--- Finding Optimal Classification Threshold ---")
y_val_pred_proba = best_xgb.predict_proba(X_val_scaled)[:, 1]
precision_curve, recall_curve, thresholds = precision_recall_curve(y_val, y_val_pred_proba)

# Calculate F1 score for each threshold
f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
print(f"Optimal threshold: {optimal_threshold:.4f}")
print(f"At optimal threshold - Precision: {precision_curve[optimal_idx]:.4f}, Recall: {recall_curve[optimal_idx]:.4f}, F1: {f1_scores[optimal_idx]:.4f}")

# Use this model for predictions with optimal threshold
model = best_xgb
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_pred_proba >= optimal_threshold).astype(int)

# Log best parameters and cross-validation scores
if mlflow_active:
    mlflow.log_params(best_params)
    mlflow.log_metric("best_cv_score", grid_search.best_score_)
    
    # Log all CV results
    cv_results = pd.DataFrame(grid_search.cv_results_)
    mlflow.log_text(cv_results.to_string(), "cv_results.txt")

# Evaluate the model
print("\n--- Evaluating Model Performance ---")

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
    mlflow.sklearn.log_model(model, "xgboost_model")
    
    # Log feature importances
    feature_importances = pd.DataFrame({
        'Feature': X_train_final.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    mlflow.log_text(feature_importances.to_string(), "feature_importances.txt")
    
    # Create and log feature importance plot
    plt.figure(figsize=(12, 10))
    top_n = 20
    top_importances = feature_importances.head(top_n)
    plt.barh(top_importances['Feature'], top_importances['Importance'])
    plt.title(f'Top {top_n} Feature Importances')
    plt.xlabel('Importance')
    plt.tight_layout()
    importance_path = "feature_importances.png"
    plt.savefig(importance_path)
    plt.close()
    
    mlflow.log_artifact(importance_path)

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

print("\n--- XGBoost Training and Evaluation Completed ---") 