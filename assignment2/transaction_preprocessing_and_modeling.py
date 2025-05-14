import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve, auc, roc_curve, roc_auc_score
from sklearn.feature_selection import RFE
import xgboost as xgb
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import dagshub
import warnings
import time
from scipy import stats
import category_encoders as ce

# Ignore warnings
warnings.filterwarnings('ignore')

# Initialize MLflow tracking
try:
    # Initialize Dagshub only if the repo info is correct
    dagshub.init(repo_owner='konstantine25b', repo_name='IEEE-CIS-Fraud-Detection', mlflow=True)
    print("DagsHub initialized successfully.")
    mlflow.set_experiment("IEEE-CIS Fraud Detection_Transaction_Only")
    print(f"MLflow experiment set to: {mlflow.get_experiment_by_name('IEEE-CIS Fraud Detection_Transaction_Only').name}")
except Exception as e:
    print(f"Could not initialize DagsHub or set MLflow experiment: {e}")
    print("Proceeding without MLflow tracking.")
    # Set a dummy client to avoid errors if tracking fails
    mlflow_active = False
else:
    mlflow_active = True

# Start MLflow run with explicit run_id
run_name = f"transaction_preprocessing_and_modeling_{time.strftime('%Y%m%d_%H%M%S')}"
if mlflow_active:
    mlflow.start_run(run_name=run_name)
    print(f"MLflow run started with name: {run_name}")

# Log start time
start_time = time.time()

# Load data
try:
    print("Loading data...")
    if mlflow_active:
        mlflow.log_param("data_source", "transaction_data_only")
    
    # Load transaction data
    transaction_df = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
    
    # Log data shape
    if mlflow_active:
        mlflow.log_param("transaction_data_shape", str(transaction_df.shape))
    print(f"Transaction data shape: {transaction_df.shape}")
except FileNotFoundError:
    print("Error: Transaction CSV file was not found. Please make sure the file path is correct.")
    if mlflow_active:
        mlflow.end_run()
    exit()

# Separate target variable
y = transaction_df['isFraud']
X = transaction_df.drop('isFraud', axis=1)

# Log class distribution
class_distribution = y.value_counts().to_dict()
if mlflow_active:
    mlflow.log_param("class_distribution", str(class_distribution))
print(f"Class distribution: {class_distribution}")

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

if mlflow_active:
    mlflow.log_param("train_test_split", "80/20 with stratification")

# Preprocessing
print("\n--- Preprocessing ---")

# Function to preprocess data
def preprocess_data(df, dataset_name="unknown"):
    # Make a copy to avoid modifying the original dataframe
    df_processed = df.copy()
    
    # Log preprocessing steps with unique key
    if mlflow_active:
        mlflow.log_param(f"preprocessing_steps_{dataset_name}", "handle_missing_values, handle_categorical, handle_numerical")
    
    # 1. Handle TransactionID - drop it as it's just an identifier
    if 'TransactionID' in df_processed.columns:
        df_processed = df_processed.drop('TransactionID', axis=1)
        if mlflow_active:
            mlflow.log_param(f"drop_columns_{dataset_name}", "TransactionID")
    
    # 2. Check null percentages for all columns
    null_percentages = (df_processed.isnull().sum() / len(df_processed)) * 100
    high_null_cols = null_percentages[null_percentages >= 60].index.tolist()
    
    # Drop columns with 60% or more nulls
    if high_null_cols:
        print(f"Dropping columns with ≥60% nulls: {high_null_cols}")
        df_processed = df_processed.drop(columns=high_null_cols)
        if mlflow_active:
            # Don't log this parameter as it's too large and causing errors
            # Instead, log the count of dropped columns
            mlflow.log_param(f"dropped_high_null_columns_count_{dataset_name}", len(high_null_cols))
    
    # 3. Handle remaining missing values
    # For numerical columns, fill with median
    numerical_cols = df_processed.select_dtypes(include=['int64', 'float64']).columns
    for col in numerical_cols:
        if df_processed[col].isnull().sum() > 0:
            median_value = df_processed[col].median()
            df_processed[col] = df_processed[col].fillna(median_value)
    
    # For categorical columns, fill with most frequent value
    categorical_cols = df_processed.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        if df_processed[col].isnull().sum() > 0:
            # Get most frequent value
            most_frequent = df_processed[col].mode()[0]
            df_processed[col] = df_processed[col].fillna(most_frequent)
    
    if mlflow_active:
        mlflow.log_param(f"missing_value_strategy_{dataset_name}", "median for numerical, most frequent for categorical")
    
    return df_processed

# Apply preprocessing with unique dataset names
print("Preprocessing training data...")
X_train_processed = preprocess_data(X_train, "train")
print("Preprocessing test data...")
X_test_processed = preprocess_data(X_test, "test")

print(f"Processed training data shape: {X_train_processed.shape}")
if mlflow_active:
    mlflow.log_param("processed_train_shape", str(X_train_processed.shape))

# After preprocessing, add this code to display null percentages before and after
print("\n--- Null Value Analysis ---")
# Calculate null percentages in original data
null_percentages_original = (X_train.isnull().sum() / len(X_train)) * 100
null_cols_original = null_percentages_original[null_percentages_original > 0].sort_values(ascending=False)

if len(null_cols_original) > 0:
    print("\nColumns with null values in original data (% nulls):")
    for col, pct in null_cols_original.items():
        print(f"{col}: {pct:.2f}%")
    
    # Log columns with high null percentages
    high_null_cols = null_cols_original[null_cols_original >= 60].index.tolist()
    if high_null_cols:
        print(f"\nColumns with ≥60% nulls (dropped): {len(high_null_cols)} columns")
        # Don't log the full list as it's too large
        if mlflow_active:
            mlflow.log_param("high_null_columns_count", len(high_null_cols))
else:
    print("No null values found in original data.")

# Check for nulls after preprocessing
null_counts_after = X_train_processed.isnull().sum()
null_cols_after = null_counts_after[null_counts_after > 0]
if len(null_cols_after) > 0:
    print("\nColumns with null values after preprocessing:")
    print(null_cols_after)
    if mlflow_active:
        mlflow.log_param("columns_with_nulls_after_preprocessing", "Yes")
else:
    print("\nNo null values found after preprocessing.")
    if mlflow_active:
        mlflow.log_param("columns_with_nulls_after_preprocessing", "None")

# Display data information
print("\nData types in processed data:")
print(X_train_processed.dtypes.value_counts())

# Display summary statistics for numerical columns
print("\nSummary statistics for numerical columns (sample of 5):")
numerical_cols = X_train_processed.select_dtypes(include=['int64', 'float64']).columns
print(X_train_processed[numerical_cols[:5]].describe())

# Display unique values for categorical columns (sample)
print("\nUnique values for categorical columns (sample of 5):")
categorical_cols = X_train_processed.select_dtypes(include=['object']).columns
for col in categorical_cols[:5]:
    unique_values = X_train_processed[col].nunique()
    print(f"{col}: {unique_values} unique values")
    # Show sample of values if not too many
    if unique_values <= 10:
        print(f"Sample values: {X_train_processed[col].unique()[:5]}")
    else:
        print(f"Sample values: {X_train_processed[col].value_counts().head(5).to_dict()}")

# Display class distribution again for reference
print("\nClass distribution:")
print(y_train.value_counts())
print(f"Fraud ratio: {y_train.mean():.4f}")

# Display categorical column information
print("\n--- Categorical Column Analysis ---")
categorical_cols = X_train_processed.select_dtypes(include=['object']).columns
print(f"Number of categorical columns: {len(categorical_cols)}")

if len(categorical_cols) > 0:
    print("\nTop 10 categorical columns by unique value count:")
    cat_unique_counts = {col: X_train_processed[col].nunique() for col in categorical_cols}
    for col, count in sorted(cat_unique_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"{col}: {count} unique values")
        # Show top 3 most frequent values
        top_values = X_train_processed[col].value_counts().head(3)
        print(f"  Top values: {dict(top_values)}")

# Feature Engineering
print("\n--- Feature Engineering ---")

# Identify categorical columns
categorical_cols = X_train_processed.select_dtypes(include=['object']).columns.tolist()
if mlflow_active:
    mlflow.log_param("categorical_columns", str(categorical_cols))
print(f"Categorical columns: {len(categorical_cols)}")

# Identify numerical columns
numerical_cols = X_train_processed.select_dtypes(include=['int64', 'float64']).columns.tolist()
if mlflow_active:
    mlflow.log_param("numerical_columns", str(len(numerical_cols)))
print(f"Numerical columns: {len(numerical_cols)}")

# Apply Weight of Evidence encoding for categorical features
print("Applying WOE encoding for categorical features...")
woe_encoder = ce.WOEEncoder(cols=categorical_cols)
X_train_woe = woe_encoder.fit_transform(X_train_processed, y_train)
X_test_woe = woe_encoder.transform(X_test_processed)

if mlflow_active:
    mlflow.log_param("encoding_method", "WOE")

# Feature Selection based on correlation
print("\n--- Feature Selection: Correlation Filter ---")

# Calculate correlation with target
correlation_with_target = pd.DataFrame()
for col in X_train_woe.columns:
    correlation = np.abs(X_train_woe[col].corr(y_train))
    correlation_with_target = pd.concat([correlation_with_target, 
                                       pd.DataFrame({'Feature': [col], 'Correlation': [correlation]})], 
                                      ignore_index=True)

# Sort by correlation
correlation_with_target = correlation_with_target.sort_values('Correlation', ascending=False)

# Select features with correlation above threshold
correlation_threshold = 0.01
selected_features_corr = correlation_with_target[correlation_with_target['Correlation'] > correlation_threshold]['Feature'].tolist()

print(f"Selected {len(selected_features_corr)} features with correlation > {correlation_threshold}")
if mlflow_active:
    mlflow.log_param("correlation_threshold", correlation_threshold)
    mlflow.log_param("features_after_correlation", len(selected_features_corr))

# Filter features based on correlation
X_train_corr = X_train_woe[selected_features_corr]
X_test_corr = X_test_woe[selected_features_corr]

# Log top correlated features
top_correlated = correlation_with_target.head(20).to_dict()
if mlflow_active:
    mlflow.log_param("top_correlated_features", str(top_correlated))

# Feature Selection using RFE
print("\n--- Feature Selection: Recursive Feature Elimination ---")

# Initialize XGBoost classifier for RFE
base_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=4,
    random_state=42,
    use_label_encoder=False,
    eval_metric='auc'
)

# Number of features to select with RFE
n_features_to_select = min(50, len(selected_features_corr))

# Initialize RFE
rfe = RFE(estimator=base_model, n_features_to_select=n_features_to_select, step=0.1)

# Fit RFE
rfe.fit(X_train_corr, y_train)

# Get selected features
selected_features_rfe = X_train_corr.columns[rfe.support_].tolist()

print(f"Selected {len(selected_features_rfe)} features with RFE")
if mlflow_active:
    mlflow.log_param("features_after_rfe", len(selected_features_rfe))
    mlflow.log_param("rfe_features", str(selected_features_rfe))

# Filter features based on RFE
X_train_rfe = X_train_corr[selected_features_rfe]
X_test_rfe = X_test_corr[selected_features_rfe]

# Model Training with XGBoost
print("\n--- Model Training: XGBoost with Hyperparameter Tuning ---")

# Define parameter grid for hyperparameter tuning
param_grid = {
    'classifier__n_estimators': [100, 200],
    'classifier__max_depth': [4, 6, 8],
    'classifier__learning_rate': [0.01, 0.05],
    'classifier__scale_pos_weight': [25, 35],  # To handle class imbalance
    'classifier__min_child_weight': [1, 3],
    'classifier__subsample': [0.8],
    'classifier__colsample_bytree': [0.8],
    'classifier__gamma': [0.1],
    'classifier__reg_alpha': [0.1],
    'classifier__reg_lambda': [1.0]
}



# Create pipeline with StandardScaler and XGBoost
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', xgb.XGBClassifier(
        objective='binary:logistic',
        use_label_encoder=False,
        eval_metric='auc',
        random_state=42,
        n_jobs=-1
    ))
])

# Set up cross-validation
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Sample a subset of data for hyperparameter tuning to speed up the process
sample_size = int(0.1 * len(X_train_rfe))  # Use 10% of data
indices = np.random.choice(len(X_train_rfe), sample_size, replace=False)
X_train_sample = X_train_rfe.iloc[indices]
y_train_sample = y_train.iloc[indices]

print(f"Using {sample_size} samples for hyperparameter tuning...")

# Set up GridSearchCV
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring='f1',  # Optimize for F1 score
    n_jobs=-1,
    verbose=1
)

# Fit GridSearchCV
grid_search.fit(X_train_sample, y_train_sample)

# Get best parameters
best_params = grid_search.best_params_
print(f"\nBest parameters: {best_params}")

# Safe MLflow logging
try:
    if mlflow_active:
        for param, value in best_params.items():
            param_name = f"best_{param.replace('classifier__', '')}"
            mlflow.log_param(param_name, value)
        mlflow.log_metric("best_cv_f1", grid_search.best_score_)
except Exception as e:
    print(f"Warning: Could not log hyperparameters to MLflow: {e}")

# Train final model with best parameters
best_xgb = xgb.XGBClassifier(
    objective='binary:logistic',
    n_estimators=best_params.get('classifier__n_estimators', 100),
    max_depth=best_params.get('classifier__max_depth', 6),
    learning_rate=best_params.get('classifier__learning_rate', 0.01),
    scale_pos_weight=best_params.get('classifier__scale_pos_weight', 30),
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

# Use a validation set for early stopping
X_train_fit, X_val, y_train_fit, y_val = train_test_split(
    X_train_rfe, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# Apply StandardScaler
scaler = StandardScaler()
X_train_fit_scaled = scaler.fit_transform(X_train_fit)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test_rfe)

# Train the model
print("\nTraining final model with best parameters...")
best_xgb.fit(
    X_train_fit_scaled, y_train_fit,
    eval_set=[(X_val_scaled, y_val)],
    verbose=True,
    early_stopping_rounds=50
)

# Find optimal threshold
print("\n--- Finding Optimal Classification Threshold ---")
y_val_pred_proba = best_xgb.predict_proba(X_val_scaled)[:, 1]
precision_curve, recall_curve, thresholds = precision_recall_curve(y_val, y_val_pred_proba)

# Calculate F1 score for each threshold
f1_scores = 2 * (precision_curve * recall_curve) / (precision_curve + recall_curve + 1e-10)
optimal_idx = np.argmax(f1_scores)
optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5

print(f"Optimal threshold: {optimal_threshold:.4f}")
print(f"At optimal threshold - Precision: {precision_curve[optimal_idx]:.4f}, Recall: {recall_curve[optimal_idx]:.4f}, F1: {f1_scores[optimal_idx]:.4f}")

# Safe MLflow logging
try:
    if mlflow_active:
        mlflow.log_param("optimal_threshold", optimal_threshold)
        mlflow.log_metric("optimal_precision", precision_curve[optimal_idx])
        mlflow.log_metric("optimal_recall", recall_curve[optimal_idx])
        mlflow.log_metric("optimal_f1", f1_scores[optimal_idx])
except Exception as e:
    print(f"Warning: Could not log threshold metrics to MLflow: {e}")

# Model Evaluation
print("\n--- Model Evaluation ---")

# Make predictions with optimal threshold
y_pred_proba = best_xgb.predict_proba(X_test_scaled)[:, 1]
y_pred = (y_pred_proba >= optimal_threshold).astype(int)

# Classification report
report = classification_report(y_test, y_pred, output_dict=True)
print(classification_report(y_test, y_pred))

# Log metrics
if mlflow_active:
    mlflow.log_metric("accuracy", report['accuracy'])
    mlflow.log_metric("precision_class_0", report['0']['precision'])
    mlflow.log_metric("recall_class_0", report['0']['recall'])
    mlflow.log_metric("f1_class_0", report['0']['f1-score'])
    mlflow.log_metric("precision_class_1", report['1']['precision'])
    mlflow.log_metric("recall_class_1", report['1']['recall'])
    mlflow.log_metric("f1_class_1", report['1']['f1-score'])

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)
if mlflow_active:
    mlflow.log_metric("roc_auc", roc_auc)

# Feature Importance
feature_importances = pd.DataFrame({
    'Feature': X_train_rfe.columns,
    'Importance': best_xgb.feature_importances_
}).sort_values('Importance', ascending=False)

print("\nTop 10 Important Features:")
print(feature_importances.head(10))

# Log feature importances
if mlflow_active:
    mlflow.log_param("top_features", str(feature_importances.head(10).to_dict()))

# Log model
if mlflow_active:
    mlflow.xgboost.log_model(best_xgb, "xgboost_model")

# Log execution time
execution_time = time.time() - start_time
print(f"\nExecution time: {execution_time:.2f} seconds")
if mlflow_active:
    mlflow.log_metric("execution_time", execution_time)

# Create and save plots
# 1. Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Not Fraud', 'Fraud'],
            yticklabels=['Not Fraud', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig('confusion_matrix.png')
if mlflow_active:
    mlflow.log_artifact('confusion_matrix.png')

# 2. ROC Curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig('roc_curve.png')
if mlflow_active:
    mlflow.log_artifact('roc_curve.png')

# 3. Precision-Recall Curve
plt.figure(figsize=(8, 6))
plt.plot(recall_curve, precision_curve, color='blue', lw=2)
plt.scatter(recall_curve[optimal_idx], precision_curve[optimal_idx], color='red', s=100, 
            label=f'Optimal threshold: {optimal_threshold:.2f}')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend()
plt.tight_layout()
plt.savefig('precision_recall_curve.png')
if mlflow_active:
    mlflow.log_artifact('precision_recall_curve.png')

# 4. Feature Importance
plt.figure(figsize=(10, 8))
sns.barplot(x='Importance', y='Feature', data=feature_importances.head(20))
plt.title('Top 20 Feature Importances')
plt.tight_layout()
plt.savefig('feature_importance.png')
if mlflow_active:
    mlflow.log_artifact('feature_importance.png')

if mlflow_active:
    print("\nMLflow tracking completed. Run ID:", mlflow.active_run().info.run_id)
    mlflow.end_run()
else:
    print("\nMLflow tracking was not active.") 