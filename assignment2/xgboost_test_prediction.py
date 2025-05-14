import pandas as pd
import numpy as np
import xgboost as xgb
import mlflow
import mlflow.xgboost
import mlflow.sklearn
import time
import warnings
import gc
import os
import dagshub
from sklearn.metrics import roc_auc_score, average_precision_score
import matplotlib.pyplot as plt
import seaborn as sns

# Ignore warnings
warnings.filterwarnings('ignore')

# Start timer
start_time = time.time()

print("=== IEEE-CIS Fraud Detection - Test Prediction ===")

# Initialize MLflow tracking
try:
    # Initialize Dagshub
    dagshub.init(repo_owner='konstantine25b', repo_name='IEEE-CIS-Fraud-Detection', mlflow=True)
    print("DagsHub initialized successfully.")
    mlflow.set_experiment("IEEE-CIS Fraud Detection_Test_Prediction")
    print(f"MLflow experiment set to: {mlflow.get_experiment_by_name('IEEE-CIS Fraud Detection_Test_Prediction').name}")
    mlflow_active = True
except Exception as e:
    print(f"Could not initialize DagsHub or set MLflow experiment: {e}")
    print("Proceeding without MLflow tracking.")
    mlflow_active = False

# Start MLflow run
run_name = f"test_prediction_{time.strftime('%Y%m%d_%H%M%S')}"
if mlflow_active:
    try:
        mlflow.start_run(run_name=run_name)
        print(f"MLflow run started with name: {run_name}")
    except Exception as e:
        print(f"Could not start MLflow run: {e}")
        mlflow_active = False

# Step 1: Load test data
print("\n--- Loading Test Data ---")

# Load test transaction data
print("Loading test transaction data...")
test_transaction = pd.read_csv('../input/ieee-fraud-detection/test_transaction.csv')
print(f"Test transaction data loaded: {test_transaction.shape}")

# Load test identity data
print("Loading test identity data...")
test_identity = pd.read_csv('../input/ieee-fraud-detection/test_identity.csv')
print(f"Test identity data loaded: {test_identity.shape}")

# Step 2: Load models and pipelines from MLflow
print("\n--- Loading Models and Pipelines from MLflow ---")

# Use the specific run ID
run_id = "e75d3cbbcd19426cbe1403e0816c2a80"
print(f"Using run ID: {run_id}")

# Load preprocessing pipelines
print("Loading preprocessing pipelines...")
identity_preprocessing_pipeline = mlflow.sklearn.load_model(
    f"runs:/{run_id}/identity_preprocessing_pipeline"
)
print("Identity preprocessing pipeline loaded")

transaction_preprocessing_pipeline = mlflow.sklearn.load_model(
    f"runs:/{run_id}/transaction_preprocessing_pipeline"
)
print("Transaction preprocessing pipeline loaded")

# Load identity flagger
identity_flagger = mlflow.sklearn.load_model(
    f"runs:/{run_id}/identity_flagger"
)
print("Identity flagger loaded")

# Load identity splitter
identity_splitter = mlflow.sklearn.load_model(
    f"runs:/{run_id}/identity_splitter"
)
print("Identity splitter loaded")

# Load identity merger
identity_merger = mlflow.sklearn.load_model(
    f"runs:/{run_id}/identity_merger"
)
print("Identity merger loaded")

# Load feature pipelines
print("Loading feature pipelines...")
with_identity_feature_pipeline = mlflow.sklearn.load_model(
    f"runs:/{run_id}/with_identity_feature_pipeline"
)
print("With identity feature pipeline loaded")

without_identity_feature_pipeline = mlflow.sklearn.load_model(
    f"runs:/{run_id}/without_identity_feature_pipeline"
)
print("Without identity feature pipeline loaded")

# Load the models
print("Loading XGBoost models...")
with_identity_model = mlflow.xgboost.load_model(
    f"runs:/{run_id}/with_identity_model"
)
print("With identity model loaded")

without_identity_model = mlflow.xgboost.load_model(
    f"runs:/{run_id}/without_identity_model"
)
print("Without identity model loaded")

# Step 3: Apply preprocessing pipelines to test data
print("\n--- Applying Preprocessing Pipelines ---")

# Standardize column names in identity data (replace 'id-X' with 'id_X')
print("Standardizing identity column names...")
identity_columns = test_identity.columns.tolist()
renamed_identity_columns = {}
for col in identity_columns:
    if col.startswith('id-'):
        renamed_identity_columns[col] = col.replace('id-', 'id_')

if renamed_identity_columns:
    print(f"Renaming {len(renamed_identity_columns)} identity columns to match training data format")
    test_identity = test_identity.rename(columns=renamed_identity_columns)
else:
    print("No identity columns need renaming")

# Preprocess identity data
print("Preprocessing identity data...")
try:
    test_identity_preprocessed = identity_preprocessing_pipeline.transform(test_identity)
    print(f"Preprocessed identity data shape: {test_identity_preprocessed.shape}")
except Exception as e:
    print(f"Error applying identity preprocessing pipeline: {e}")
    print("Falling back to manual preprocessing...")
    identity_null_percentage = test_identity.isnull().mean() * 100
    identity_high_null_cols = identity_null_percentage[identity_null_percentage >= 20].index.tolist()
    test_identity_preprocessed = test_identity.drop(columns=identity_high_null_cols, errors='ignore')
    print(f"Manually preprocessed identity data shape: {test_identity_preprocessed.shape}")

# Standardize column names in transaction data if needed
print("Standardizing transaction column names...")
transaction_columns = test_transaction.columns.tolist()
renamed_transaction_columns = {}
for col in transaction_columns:
    if col.startswith('id-'):
        renamed_transaction_columns[col] = col.replace('id-', 'id_')
    # Add other column name standardizations if needed

if renamed_transaction_columns:
    print(f"Renaming {len(renamed_transaction_columns)} transaction columns to match training data format")
    test_transaction = test_transaction.rename(columns=renamed_transaction_columns)
else:
    print("No transaction columns need renaming")

# Preprocess transaction data
print("Preprocessing transaction data...")
try:
    test_transaction_preprocessed = transaction_preprocessing_pipeline.transform(test_transaction)
    print(f"Preprocessed transaction data shape: {test_transaction_preprocessed.shape}")
except Exception as e:
    print(f"Error applying transaction preprocessing pipeline: {e}")
    print("Falling back to manual preprocessing...")
    transaction_null_percentage = test_transaction.isnull().mean() * 100
    transaction_high_null_cols = transaction_null_percentage[transaction_null_percentage >= 60].index.tolist()
    test_transaction_preprocessed = test_transaction.drop(columns=transaction_high_null_cols, errors='ignore')
    print(f"Manually preprocessed transaction data shape: {test_transaction_preprocessed.shape}")

# Add identity flag
print("Adding identity flag...")
try:
    # Try using the identity flagger
    test_transaction_with_flag = identity_flagger.transform(test_transaction_preprocessed)
    print(f"Added has_identity flag. Transactions with identity: {test_transaction_with_flag['has_identity'].sum()}")
    
    # If the flagger didn't find any matches, add the flag manually
    if test_transaction_with_flag['has_identity'].sum() == 0:
        print("Identity flagger found no matches. Adding flag manually...")
        test_transaction_with_flag['has_identity'] = test_transaction_with_flag['TransactionID'].isin(
            test_identity_preprocessed['TransactionID']).astype(int)
        print(f"Added identity flag manually. Transactions with identity: {test_transaction_with_flag['has_identity'].sum()}")
except Exception as e:
    print(f"Error applying identity flagger: {e}")
    print("Adding identity flag manually...")
    test_transaction_with_flag = test_transaction_preprocessed.copy()
    test_transaction_with_flag['has_identity'] = test_transaction_with_flag['TransactionID'].isin(
        test_identity_preprocessed['TransactionID']).astype(int)
    print(f"Added identity flag manually. Transactions with identity: {test_transaction_with_flag['has_identity'].sum()}")

# Split data based on identity presence
print("Splitting data based on identity presence...")
try:
    # Since we've manually added the flag, let's use the manual approach for consistency
    test_with_identity = test_transaction_with_flag[test_transaction_with_flag['has_identity'] == 1].copy()
    test_without_identity = test_transaction_with_flag[test_transaction_with_flag['has_identity'] == 0].copy()
    print(f"Split data manually. WITH identity: {test_with_identity.shape}, WITHOUT identity: {test_without_identity.shape}")
except Exception as e:
    print(f"Error splitting data: {e}")
    print("Falling back to manual splitting...")
    test_with_identity = test_transaction_with_flag[test_transaction_with_flag['has_identity'] == 1].copy()
    test_without_identity = test_transaction_with_flag[test_transaction_with_flag['has_identity'] == 0].copy()
    print(f"Split data manually. WITH identity: {test_with_identity.shape}, WITHOUT identity: {test_without_identity.shape}")

# Merge identity data with transactions that have identity
print("Merging identity data...")
try:
    # Try using the identity merger with just one argument
    test_with_identity_merged = identity_merger.transform(test_with_identity)
    print(f"Merged data using identity merger. Shape: {test_with_identity_merged.shape}")
except Exception as e:
    print(f"Error applying identity merger: {e}")
    print("Merging data manually...")
    test_with_identity_merged = pd.merge(test_with_identity, test_identity_preprocessed, 
                                         on='TransactionID', how='left')
    print(f"Merged data manually. Shape: {test_with_identity_merged.shape}")

# Step 4: Prepare features for prediction
print("\n--- Preparing Features for Prediction ---")

# Extract TransactionID for final submission
transaction_ids_with_identity = test_with_identity['TransactionID'].values
transaction_ids_without_identity = test_without_identity['TransactionID'].values

# Prepare features for WITH identity data
print("Preparing features for WITH identity data...")
X_with_identity_test = test_with_identity_merged.drop(columns=['TransactionID', 'has_identity'], errors='ignore')
print(f"WITH identity features shape: {X_with_identity_test.shape}")

# Prepare features for WITHOUT identity data
print("Preparing features for WITHOUT identity data...")
X_without_identity_test = test_without_identity.drop(columns=['TransactionID', 'has_identity'], errors='ignore')
print(f"WITHOUT identity features shape: {X_without_identity_test.shape}")

# Step 5: Apply feature pipelines
print("\n--- Applying Feature Pipelines ---")

# Apply WITH identity feature pipeline
print("Applying WITH identity feature pipeline...")
try:
    X_with_identity_test_processed = with_identity_feature_pipeline.transform(X_with_identity_test)
    print(f"Processed WITH identity features shape: {X_with_identity_test_processed.shape}")
except Exception as e:
    print(f"Error applying WITH identity feature pipeline: {e}")
    print("Falling back to manual feature processing...")
    
    # Fill missing values
    X_with_identity_test = X_with_identity_test.fillna(-999)
    
    # Convert categorical columns to numeric
    for col in X_with_identity_test.select_dtypes(include=['object']).columns:
        X_with_identity_test[col] = pd.factorize(X_with_identity_test[col])[0]
    
    X_with_identity_test_processed = X_with_identity_test.values
    print(f"Manually processed WITH identity features shape: {X_with_identity_test_processed.shape}")

# Apply WITHOUT identity feature pipeline
print("Applying WITHOUT identity feature pipeline...")
try:
    X_without_identity_test_processed = without_identity_feature_pipeline.transform(X_without_identity_test)
    print(f"Processed WITHOUT identity features shape: {X_without_identity_test_processed.shape}")
except Exception as e:
    print(f"Error applying WITHOUT identity feature pipeline: {e}")
    print("Falling back to manual feature processing...")
    
    # Fill missing values
    X_without_identity_test = X_without_identity_test.fillna(-999)
    
    # Convert categorical columns to numeric
    for col in X_without_identity_test.select_dtypes(include=['object']).columns:
        X_without_identity_test[col] = pd.factorize(X_without_identity_test[col])[0]
    
    X_without_identity_test_processed = X_without_identity_test.values
    print(f"Manually processed WITHOUT identity features shape: {X_without_identity_test_processed.shape}")

# Step 6: Create DMatrix objects
print("\n--- Creating DMatrix Objects ---")

dtest_with_identity = xgb.DMatrix(X_with_identity_test_processed)
print(f"DMatrix WITH identity created")

dtest_without_identity = xgb.DMatrix(X_without_identity_test_processed)
print(f"DMatrix WITHOUT identity created")

# Step 7: Make predictions
print("\n--- Making Predictions ---")

# Make predictions for WITH identity data
print("Generating predictions for WITH identity data...")
with_identity_preds = with_identity_model.predict(dtest_with_identity)
print(f"WITH identity predictions shape: {with_identity_preds.shape}")

# Make predictions for WITHOUT identity data
print("Generating predictions for WITHOUT identity data...")
without_identity_preds = without_identity_model.predict(dtest_without_identity)
print(f"WITHOUT identity predictions shape: {without_identity_preds.shape}")

# Step 8: Create submission file
print("\n--- Creating Submission File ---")

# Create separate DataFrames for each prediction set
with_identity_submission = pd.DataFrame({
    'TransactionID': transaction_ids_with_identity,
    'isFraud': with_identity_preds
})

without_identity_submission = pd.DataFrame({
    'TransactionID': transaction_ids_without_identity,
    'isFraud': without_identity_preds
})

# Combine the predictions
submission = pd.concat([with_identity_submission, without_identity_submission])

# Sort by TransactionID to maintain original order
submission = submission.sort_values('TransactionID')

# Save submission file
submission_file = 'xgboost_submission.csv'
submission.to_csv(submission_file, index=False)
print(f"Submission file saved to {submission_file}")
print(f"Final submission shape: {submission.shape}")

if mlflow_active:
    mlflow.log_artifact(submission_file)
    print(f"Submission file logged to MLflow")

# Log execution time
execution_time = time.time() - start_time
print(f"\nTotal execution time: {execution_time:.2f} seconds")

if mlflow_active:
    mlflow.log_metric("execution_time", execution_time)
    mlflow.end_run()
    print("MLflow run completed successfully.")

print("\n--- Test Prediction Complete ---") 