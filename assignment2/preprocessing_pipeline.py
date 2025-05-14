import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from category_encoders import WOEEncoder
from sklearn.base import BaseEstimator, TransformerMixin
import mlflow
import mlflow.sklearn
import dagshub
import time
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Import experimental features in the correct order
from sklearn.experimental import enable_iterative_imputer  # This must come first
from sklearn.impute import KNNImputer, IterativeImputer
from sklearn.linear_model import BayesianRidge

# Initialize MLflow tracking
try:
    # Initialize Dagshub only if the repo info is correct
    dagshub.init(repo_owner='konstantine25b', repo_name='IEEE-CIS-Fraud-Detection', mlflow=True)
    print("DagsHub initialized successfully.")
    mlflow.set_experiment("IEEE-CIS Fraud Detection_PreProcessing")
    print(f"MLflow experiment set to: {mlflow.get_experiment_by_name('IEEE-CIS Fraud Detection_PreProcessing').name}")
except Exception as e:
    print(f"Could not initialize DagsHub or set MLflow experiment: {e}")
    print("Proceeding without MLflow tracking.")
    # Set a dummy client to avoid errors if tracking fails
    mlflow_active = False
else:
    mlflow_active = True

# Custom transformer to remove columns with high missing values
class HighNARemover(BaseEstimator, TransformerMixin):
    def __init__(self, threshold):
        self.threshold = threshold
        self.columns_to_drop = None
        
    def fit(self, X, y=None):
        # Calculate percentage of missing values for each column
        na_percentage = X.isna().mean()
        # Identify columns with missing values above threshold
        self.columns_to_drop = na_percentage[na_percentage > self.threshold].index.tolist()
        return self
    
    def transform(self, X):
        # Drop columns with high missing values
        return X.drop(columns=self.columns_to_drop, errors='ignore')
    
    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return []
        return [f for f in input_features if f not in self.columns_to_drop]

# Start MLflow run with explicit run_id
run_name = f"preprocessing_pipeline_{time.strftime('%Y%m%d_%H%M%S')}"
if mlflow_active:
    mlflow.start_run(run_name=run_name)
    print(f"MLflow run started with name: {run_name}")

# Load data
try:
    identity_df = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_identity.csv')
    transaction_df = pd.read_csv('/kaggle/input/ieee-fraud-detection/train_transaction.csv')
except FileNotFoundError:
    print("Error: One or both of the CSV files were not found. Please make sure the file paths are correct.")
    if mlflow_active:
        mlflow.end_run()
    exit()

# After loading data, print initial information
print(f"Loaded identity data shape: {identity_df.shape}")
print(f"Loaded transaction data shape: {transaction_df.shape}")
print(f"Number of unique TransactionIDs in identity data: {identity_df['TransactionID'].nunique()}")
print(f"Number of unique TransactionIDs in transaction data: {transaction_df['TransactionID'].nunique()}")
print(f"Target variable distribution: \n{transaction_df['isFraud'].value_counts(normalize=True)}")

# Log initial data metrics
if mlflow_active:
    mlflow.log_param("identity_initial_rows", identity_df.shape[0])
    mlflow.log_param("identity_initial_cols", identity_df.shape[1])
    mlflow.log_param("transaction_initial_rows", transaction_df.shape[0])
    mlflow.log_param("transaction_initial_cols", transaction_df.shape[1])
    mlflow.log_param("fraud_rate", float(transaction_df['isFraud'].mean()))

# First, remove high NA columns from both datasets
identity_na_remover = HighNARemover(threshold=0.2)
transaction_na_remover = HighNARemover(threshold=0.6)

# Fit and transform to remove high NA columns
identity_df_filtered = identity_na_remover.fit_transform(identity_df)
transaction_df_filtered = transaction_na_remover.fit_transform(transaction_df)

print("\n--- After Removing High NA Columns ---")
print(f"Identity filtered shape: {identity_df_filtered.shape}")
print(f"Transaction filtered shape: {transaction_df_filtered.shape}")
print(f"Columns dropped from identity: {len(identity_na_remover.columns_to_drop)}")
print(f"Columns dropped from transaction: {len(transaction_na_remover.columns_to_drop)}")

# Log NA removal metrics
if mlflow_active:
    mlflow.log_param("identity_na_threshold", 0.2)
    mlflow.log_param("transaction_na_threshold", 0.6)
    mlflow.log_param("identity_cols_dropped", len(identity_na_remover.columns_to_drop))
    mlflow.log_param("transaction_cols_dropped", len(transaction_na_remover.columns_to_drop))
    mlflow.log_param("identity_cols_remaining", identity_df_filtered.shape[1])
    mlflow.log_param("transaction_cols_remaining", transaction_df_filtered.shape[1])
    
    # Log the names of dropped columns
    mlflow.log_text("\n".join(identity_na_remover.columns_to_drop), "identity_dropped_columns.txt")
    mlflow.log_text("\n".join(transaction_na_remover.columns_to_drop), "transaction_dropped_columns.txt")

# Now split the transaction data
X_transaction = transaction_df_filtered.drop('isFraud', axis=1)
y_transaction = transaction_df_filtered['isFraud']

X_transaction_train, X_transaction_test, y_train, y_test = train_test_split(
    X_transaction, y_transaction, test_size=0.2, random_state=42, stratify=y_transaction
)

# Get the corresponding identity records for train and test sets
train_transaction_ids = X_transaction_train['TransactionID'].values
test_transaction_ids = X_transaction_test['TransactionID'].values

# Filter identity dataframe based on TransactionIDs
X_identity_train = identity_df_filtered[identity_df_filtered['TransactionID'].isin(train_transaction_ids)]
X_identity_test = identity_df_filtered[identity_df_filtered['TransactionID'].isin(test_transaction_ids)]

# Create a mapping from TransactionID to target for identity data
transaction_id_to_target = dict(zip(transaction_df_filtered['TransactionID'], transaction_df_filtered['isFraud']))
y_identity_train = X_identity_train['TransactionID'].map(transaction_id_to_target)
y_identity_test = X_identity_test['TransactionID'].map(transaction_id_to_target)

# After splitting data
print("\n--- Train-Test Split Information ---")
print(f"Transaction train set shape: {X_transaction_train.shape}")
print(f"Transaction test set shape: {X_transaction_test.shape}")
print(f"Identity train set shape: {X_identity_train.shape}")
print(f"Identity test shape: {X_identity_test.shape}")
print(f"Transaction target train distribution: \n{y_train.value_counts(normalize=True)}")
print(f"Identity target train distribution: \n{y_identity_train.value_counts(normalize=True)}")

# Log train-test split metrics
if mlflow_active:
    mlflow.log_param("test_size", 0.2)
    mlflow.log_param("random_state", 42)
    mlflow.log_param("transaction_train_rows", X_transaction_train.shape[0])
    mlflow.log_param("transaction_test_rows", X_transaction_test.shape[0])
    mlflow.log_param("identity_train_rows", X_identity_train.shape[0])
    mlflow.log_param("identity_test_rows", X_identity_test.shape[0])
    mlflow.log_param("train_fraud_rate", float(y_train.mean()))
    mlflow.log_param("test_fraud_rate", float(y_test.mean()))

# Function to categorize columns
def categorize_columns(df):
    # Identify numeric and categorical columns
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # For categorical columns, separate those with ≤4 unique values and >4 unique values
    categorical_low_cardinality = []
    categorical_high_cardinality = []
    
    for col in categorical_cols:
        if df[col].nunique() <= 4:
            categorical_low_cardinality.append(col)
        else:
            categorical_high_cardinality.append(col)
    
    # Remove 'TransactionID' from numeric columns if present
    if 'TransactionID' in numeric_cols:
        numeric_cols.remove('TransactionID')
    
    return numeric_cols, categorical_low_cardinality, categorical_high_cardinality

# Now categorize columns
identity_numeric_cols, identity_cat_low, identity_cat_high = categorize_columns(X_identity_train)
transaction_numeric_cols, transaction_cat_low, transaction_cat_high = categorize_columns(X_transaction_train)

# After categorizing columns
print("\n--- Column Categorization ---")
print(f"Identity numeric columns: {len(identity_numeric_cols)}")
print(f"Identity low cardinality categorical columns: {len(identity_cat_low)}")
print(f"Identity high cardinality categorical columns: {len(identity_cat_high)}")
print(f"Transaction numeric columns: {len(transaction_numeric_cols)}")
print(f"Transaction low cardinality categorical columns: {len(transaction_cat_low)}")
print(f"Transaction high cardinality categorical columns: {len(transaction_cat_high)}")

# Log column categorization metrics
if mlflow_active:
    mlflow.log_param("identity_numeric_cols", len(identity_numeric_cols))
    mlflow.log_param("identity_cat_low_cols", len(identity_cat_low))
    mlflow.log_param("identity_cat_high_cols", len(identity_cat_high))
    mlflow.log_param("transaction_numeric_cols", len(transaction_numeric_cols))
    mlflow.log_param("transaction_cat_low_cols", len(transaction_cat_low))
    mlflow.log_param("transaction_cat_high_cols", len(transaction_cat_high))
    
    # Log column names by category
    mlflow.log_text("\n".join(identity_numeric_cols), "identity_numeric_columns.txt")
    mlflow.log_text("\n".join(identity_cat_low), "identity_cat_low_columns.txt")
    mlflow.log_text("\n".join(identity_cat_high), "identity_cat_high_columns.txt")
    mlflow.log_text("\n".join(transaction_numeric_cols), "transaction_numeric_columns.txt")
    mlflow.log_text("\n".join(transaction_cat_low), "transaction_cat_low_columns.txt")
    mlflow.log_text("\n".join(transaction_cat_high), "transaction_cat_high_columns.txt")

# Create identity preprocessing pipeline
identity_preprocessor = Pipeline([
    ('column_transformer', ColumnTransformer([
        ('num_imputer', Pipeline([
            ('imputer', SimpleImputer(strategy='mean'))
        ]), identity_numeric_cols),
        ('cat_low_encoder', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), identity_cat_low),
        ('cat_high_encoder', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('woe', WOEEncoder(handle_missing='return_nan'))
        ]), identity_cat_high)
    ], remainder='drop'))
])

# Create transaction preprocessing pipeline
transaction_preprocessor = Pipeline([
    ('column_transformer', ColumnTransformer([
        ('num_imputer', Pipeline([
            ('imputer', SimpleImputer(strategy='mean'))
        ]), transaction_numeric_cols),
        ('cat_low_encoder', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), transaction_cat_low),
        ('cat_high_encoder', Pipeline([
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('woe', WOEEncoder(handle_missing='return_nan'))
        ]), transaction_cat_high)
    ], remainder='drop'))
])

# Log pipeline parameters
if mlflow_active:
    mlflow.log_param("numeric_imputation_strategy", "mean")
    mlflow.log_param("categorical_imputation_strategy", "most_frequent")
    mlflow.log_param("low_cardinality_encoding", "one-hot")
    mlflow.log_param("high_cardinality_encoding", "weight-of-evidence")

# Fit and transform the identity data
start_time = time.time()
identity_preprocessor.fit(X_identity_train, y_identity_train)
X_identity_train_processed = identity_preprocessor.transform(X_identity_train)
X_identity_test_processed = identity_preprocessor.transform(X_identity_test)
identity_processing_time = time.time() - start_time

# Fit and transform the transaction data
start_time = time.time()
transaction_preprocessor.fit(X_transaction_train, y_train)
X_transaction_train_processed = transaction_preprocessor.transform(X_transaction_train)
X_transaction_test_processed = transaction_preprocessor.transform(X_transaction_test)
transaction_processing_time = time.time() - start_time

# After transformation, print more detailed information
print("\n--- Processed Data Information ---")
print(f"Processed identity train data shape: {X_identity_train_processed.shape}")
print(f"Processed identity test data shape: {X_identity_test_processed.shape}")
print(f"Processed transaction train data shape: {X_transaction_train_processed.shape}")
print(f"Processed transaction test data shape: {X_transaction_test_processed.shape}")
print(f"Identity processing time: {identity_processing_time:.2f} seconds")
print(f"Transaction processing time: {transaction_processing_time:.2f} seconds")

# Log processing metrics
if mlflow_active:
    mlflow.log_metric("identity_processing_time", identity_processing_time)
    mlflow.log_metric("transaction_processing_time", transaction_processing_time)
    mlflow.log_param("identity_processed_features", X_identity_train_processed.shape[1])
    mlflow.log_param("transaction_processed_features", X_transaction_train_processed.shape[1])

# Check for any remaining NaN values in processed data
print("\n--- Checking for Remaining NaN Values ---")
if isinstance(X_identity_train_processed, np.ndarray):
    identity_nan_count = np.isnan(X_identity_train_processed).sum()
    transaction_nan_count = np.isnan(X_transaction_train_processed).sum()
    print(f"Identity train NaN count: {identity_nan_count}")
    print(f"Transaction train NaN count: {transaction_nan_count}")
else:
    identity_nan_count = X_identity_train_processed.isna().sum().sum()
    transaction_nan_count = X_transaction_train_processed.isna().sum().sum()
    print(f"Identity train NaN count: {identity_nan_count}")
    print(f"Transaction train NaN count: {transaction_nan_count}")

# Log NaN counts
if mlflow_active:
    mlflow.log_metric("identity_remaining_nan_count", float(identity_nan_count))
    mlflow.log_metric("transaction_remaining_nan_count", float(transaction_nan_count))

# Create wrapper classes that add predict methods to our pipelines
class IdentityPipelineWrapper(Pipeline):
    def predict(self, X):
        """Add a predict method that just returns the transformed data.
        This allows MLflow to log the model with python_function flavor."""
        return self.transform(X)

class TransactionPipelineWrapper(Pipeline):
    def predict(self, X):
        """Add a predict method that just returns the transformed data.
        This allows MLflow to log the model with python_function flavor."""
        return self.transform(X)

# Create wrapped versions of our pipelines
identity_pipeline_wrapped = IdentityPipelineWrapper(
    steps=[
        ('na_remover', identity_na_remover),
        ('preprocessor', identity_preprocessor)
    ]
)

transaction_pipeline_wrapped = TransactionPipelineWrapper(
    steps=[
        ('na_remover', transaction_na_remover),
        ('preprocessor', transaction_preprocessor)
    ]
)

# Create proper input examples for model signatures with some missing values
identity_example = X_identity_train.iloc[:5].copy()
# Introduce some NaN values to ensure proper schema inference
for col in identity_example.select_dtypes(include=['int64']).columns[:2]:
    identity_example.loc[0, col] = np.nan

transaction_example = X_transaction_train.iloc[:5].copy()
# Introduce some NaN values to ensure proper schema inference
for col in transaction_example.select_dtypes(include=['int64']).columns[:2]:
    transaction_example.loc[0, col] = np.nan

# Log the wrapped models with input examples
if mlflow_active:
    # Log identity pipeline with signature
    mlflow.sklearn.log_model(
        identity_pipeline_wrapped, 
        "identity_pipeline_model",
        input_example=identity_example
    )
    
    # Log transaction pipeline with signature
    mlflow.sklearn.log_model(
        transaction_pipeline_wrapped, 
        "transaction_pipeline_model",
        input_example=transaction_example
    )
    
    # Log sample data and statistics as before
    mlflow.log_text(str(X_identity_train.iloc[0].to_dict()), "identity_sample_input.json")
    mlflow.log_text(str(X_transaction_train.iloc[0].to_dict()), "transaction_sample_input.json")
    mlflow.log_text(pd.DataFrame(X_identity_train_processed).describe().to_string(), "identity_processed_stats.txt")
    mlflow.log_text(pd.DataFrame(X_transaction_train_processed).describe().to_string(), "transaction_processed_stats.txt")
    
    # End the MLflow run
    mlflow.end_run()
    print("MLflow run completed and artifacts logged.")

# After processing both datasets separately, merge them based on TransactionID
# Add this code before ending the MLflow run

# Merge the processed datasets
print("\n--- Merging Processed Datasets ---")
# First, add TransactionID back to the processed data
X_transaction_train_with_id = pd.DataFrame(X_transaction_train_processed)
X_transaction_train_with_id['TransactionID'] = X_transaction_train['TransactionID'].values

X_transaction_test_with_id = pd.DataFrame(X_transaction_test_processed)
X_transaction_test_with_id['TransactionID'] = X_transaction_test['TransactionID'].values

X_identity_train_with_id = pd.DataFrame(X_identity_train_processed)
X_identity_train_with_id['TransactionID'] = X_identity_train['TransactionID'].values

X_identity_test_with_id = pd.DataFrame(X_identity_test_processed)
X_identity_test_with_id['TransactionID'] = X_identity_test['TransactionID'].values

# Merge train datasets
merged_train = pd.merge(
    X_transaction_train_with_id,
    X_identity_train_with_id,
    on='TransactionID',
    how='left'
)

# Merge test datasets
merged_test = pd.merge(
    X_transaction_test_with_id,
    X_identity_test_with_id,
    on='TransactionID',
    how='left'
)

# Remove TransactionID from merged datasets
merged_train_final = merged_train.drop('TransactionID', axis=1)
merged_test_final = merged_test.drop('TransactionID', axis=1)

# Check for NAs in merged datasets
print("\n--- Checking for NAs in Merged Datasets ---")
train_na_count = merged_train_final.isna().sum().sum()
test_na_count = merged_test_final.isna().sum().sum()
print(f"Merged train dataset shape: {merged_train_final.shape}")
print(f"Merged test dataset shape: {merged_test_final.shape}")
print(f"NAs in merged train dataset: {train_na_count}")
print(f"NAs in merged test dataset: {test_na_count}")

# After merging datasets, add this code to analyze NA distribution by column

# Check NA distribution by column
print("\n--- NA Distribution in Merged Train Dataset ---")
na_percentages = merged_train_final.isna().mean().sort_values(ascending=False)
print("Top 20 columns with highest NA percentages:")
print(na_percentages.head(20))

# Count columns with different NA percentage ranges
na_ranges = {
    "100%": sum(na_percentages == 1.0),
    "90-100%": sum((na_percentages >= 0.9) & (na_percentages < 1.0)),
    "75-90%": sum((na_percentages >= 0.75) & (na_percentages < 0.9)),
    "50-75%": sum((na_percentages >= 0.5) & (na_percentages < 0.75)),
    "25-50%": sum((na_percentages >= 0.25) & (na_percentages < 0.5)),
    "10-25%": sum((na_percentages >= 0.1) & (na_percentages < 0.25)),
    "0-10%": sum((na_percentages > 0) & (na_percentages < 0.1)),
    "0%": sum(na_percentages == 0)
}

print("\nNA percentage distribution across columns:")
for range_name, count in na_ranges.items():
    print(f"{range_name}: {count} columns")

# Alternative approach: identify columns that have high NA percentages
# (since identity columns will have NAs for transactions without identity data)
identity_columns = na_percentages[na_percentages > 0.7].index.tolist()

print(f"\nNumber of columns from identity dataset (identified by NA percentage > 70%): {len(identity_columns)}")
print(f"Average NA percentage in identity columns: {merged_train_final[identity_columns].isna().mean().mean() * 100:.2f}%")

# After analyzing the NA distribution, let's handle the identity columns more appropriately

print("\n--- Handling Identity Data NAs with Simple Imputation ---")
print("Using simple imputation methods:")
print("1. Mean imputation for numeric features")
print("2. Mode imputation for categorical features")

# Separate numeric and categorical columns
merged_numeric_cols = merged_train_final.select_dtypes(include=['int64', 'float64']).columns
merged_object_cols = merged_train_final.select_dtypes(include=['object']).columns

# First, add the identity flag (this should be done before imputation)
identity_flag_col = 'has_identity_data'
merged_train_final[identity_flag_col] = (~merged_train_final[identity_columns].isna().all(axis=1)).astype(int)
merged_test_final[identity_flag_col] = (~merged_test_final[identity_columns].isna().all(axis=1)).astype(int)

print(f"Added '{identity_flag_col}' column to indicate presence of identity data")
print(f"Transactions with identity data in train set: {merged_train_final[identity_flag_col].mean() * 100:.2f}%")
print(f"Transactions with identity data in test set: {merged_test_final[identity_flag_col].mean() * 100:.2f}%")

# For numeric columns, use mean imputation
print("\nApplying mean imputation for numeric columns...")
numeric_cols_to_impute = [col for col in merged_numeric_cols if col != identity_flag_col]
if numeric_cols_to_impute:
    # Calculate means from train data
    column_means = merged_train_final[numeric_cols_to_impute].mean()
    
    # Apply mean imputation to both train and test
    for col in numeric_cols_to_impute:
        merged_train_final[col] = merged_train_final[col].fillna(column_means[col])
        merged_test_final[col] = merged_test_final[col].fillna(column_means[col])

# For categorical columns, use mode imputation
print("Applying mode imputation for categorical columns...")
if len(merged_object_cols) > 0:
    for col in merged_object_cols:
        # Get the most frequent value from the train set
        most_frequent = merged_train_final[col].mode()[0]
        # Fill NAs in both train and test
        merged_train_final[col] = merged_train_final[col].fillna(most_frequent)
        merged_test_final[col] = merged_test_final[col].fillna(most_frequent)

# Verify NAs are gone
train_na_count_after = merged_train_final.isna().sum().sum()
test_na_count_after = merged_test_final.isna().sum().sum()
print(f"NAs after imputation - train: {train_na_count_after}")
print(f"NAs after imputation - test: {test_na_count_after}")

# If there are still NAs (which shouldn't happen), fill them with 0 as a fallback
if train_na_count_after > 0 or test_na_count_after > 0:
    print("Warning: Some NAs remain after imputation. Filling remaining NAs with 0...")
    merged_train_final = merged_train_final.fillna(0)
    merged_test_final = merged_test_final.fillna(0)

# Log imputation information
if mlflow_active:
    mlflow.log_param("numeric_imputation_method", "mean")
    mlflow.log_param("categorical_imputation_method", "mode")
    mlflow.log_metric("train_na_count_after_imputation", float(train_na_count_after))
    mlflow.log_metric("test_na_count_after_imputation", float(test_na_count_after))

# Log merged dataset metrics
if mlflow_active:
    mlflow.log_param("merged_train_rows", merged_train_final.shape[0])
    mlflow.log_param("merged_train_cols", merged_train_final.shape[1])
    mlflow.log_param("merged_test_rows", merged_test_final.shape[0])
    mlflow.log_param("merged_test_cols", merged_test_final.shape[1])
    mlflow.log_metric("merged_train_na_count", float(train_na_count))
    mlflow.log_metric("merged_test_na_count", float(test_na_count))
    
    # Log sample of merged data
    mlflow.log_text(merged_train_final.head().to_string(), "merged_train_sample.txt")
    mlflow.log_text(merged_train_final.describe().to_string(), "merged_train_stats.txt")

print("\n--- Preprocessing completed successfully ---")
print("The preprocessing pipelines have been logged to MLflow and can be retrieved for future use.")

# After the preprocessing is completed, start a new MLflow run for feature selection
if mlflow_active:
    # End the previous run if it's still active
    try:
        mlflow.end_run()
    except:
        pass
    
    # Start a new run for feature selection
    feature_selection_run_name = f"feature_selection_{time.strftime('%Y%m%d_%H%M%S')}"
    mlflow.start_run(run_name=feature_selection_run_name)
    print(f"MLflow run started for feature selection with name: {feature_selection_run_name}")

# Step 1: Remove highly correlated features
print("\n--- Removing Highly Correlated Features ---")
print("Calculating correlation matrix...")

# Calculate correlation matrix and handle NaNs from the beginning
correlation_matrix = merged_train_final.corr().abs().fillna(0)

# Create a mask for the upper triangle
mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))

# No need for a separate correlation_matrix_viz since we've already handled NaNs
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, mask=mask, cmap='coolwarm', center=0, linewidths=0.5)

# Find features with correlation greater than 0.95
to_drop = []
correlation_threshold = 0.94

# Find highly correlated features
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
            colname = correlation_matrix.columns[i]
            to_drop.append(colname)
            break

# Remove highly correlated features
print(f"Removing {len(to_drop)} highly correlated features...")
merged_train_final_uncorrelated = merged_train_final.drop(columns=to_drop)
merged_test_final_uncorrelated = merged_test_final.drop(columns=to_drop)

print(f"Dataset shape after correlation removal - train: {merged_train_final_uncorrelated.shape}")
print(f"Dataset shape after correlation removal - test: {merged_test_final_uncorrelated.shape}")

# Log correlation removal metrics with unique parameter names
if mlflow_active:
    mlflow.log_param("fs_correlation_threshold", correlation_threshold)
    mlflow.log_param("fs_features_removed_by_correlation", len(to_drop))
    mlflow.log_text("\n".join(map(str, to_drop)), "removed_correlated_features.txt")

# After removing highly correlated features
print("\n--- Highly Correlated Features Removed ---")
print(f"Number of features removed: {len(to_drop)}")
print("Top 20 removed features:")
print("\n".join(map(str, to_drop[:20])))
if len(to_drop) > 20:
    print(f"... and {len(to_drop) - 20} more")

# Save the list of removed features to a CSV file
pd.Series(to_drop, name="removed_features").to_csv("removed_correlated_features.csv", index=False)
print("Full list of removed features saved to 'removed_correlated_features.csv'")

# After correlation removal, add this code to analyze the source of remaining features
print("\n--- Analysis of Features After Correlation Removal ---")

# Count how many remaining features came from identity dataset
remaining_identity_features = [feat for feat in merged_train_final_uncorrelated.columns if feat in identity_columns]
remaining_transaction_features = [feat for feat in merged_train_final_uncorrelated.columns 
                                 if feat not in identity_columns and feat != identity_flag_col]

# Count the identity flag separately
has_identity_flag = identity_flag_col in merged_train_final_uncorrelated.columns

print(f"Features after correlation removal: {merged_train_final_uncorrelated.shape[1]}")
print(f"Remaining identity features: {len(remaining_identity_features)} ({len(remaining_identity_features)/merged_train_final_uncorrelated.shape[1]*100:.1f}%)")
print(f"Remaining transaction features: {len(remaining_transaction_features)} ({len(remaining_transaction_features)/merged_train_final_uncorrelated.shape[1]*100:.1f}%)")
print(f"Identity flag present: {has_identity_flag}")

# Log this information to MLflow
if mlflow_active:
    mlflow.log_param("fs_identity_features_after_correlation", len(remaining_identity_features))
    mlflow.log_param("fs_transaction_features_after_correlation", len(remaining_transaction_features))
    mlflow.log_metric("fs_identity_features_after_correlation_pct", 
                     len(remaining_identity_features)/merged_train_final_uncorrelated.shape[1]*100)
    mlflow.log_metric("fs_transaction_features_after_correlation_pct", 
                     len(remaining_transaction_features)/merged_train_final_uncorrelated.shape[1]*100)
    
    # Log lists of remaining features by source
    mlflow.log_text("\n".join(remaining_identity_features), "remaining_identity_features_after_correlation.txt")
    mlflow.log_text("\n".join(map(str, remaining_transaction_features)), "remaining_transaction_features_after_correlation.txt")
    
    # Create and log a pie chart of feature sources after correlation removal
    plt.figure(figsize=(8, 8))
    plt.pie([len(remaining_identity_features), len(remaining_transaction_features), int(has_identity_flag)], 
            labels=['Identity Features', 'Transaction Features', 'Identity Flag'],
            autopct='%1.1f%%',
            colors=['#ff9999','#66b3ff', '#99ff99'],
            explode=(0.1, 0, 0))
    plt.title('Feature Sources After Correlation Removal')
    plt.tight_layout()
    
    # Save and log the pie chart
    correlation_feature_chart_path = "feature_sources_after_correlation.png"
    plt.savefig(correlation_feature_chart_path)
    mlflow.log_artifact(correlation_feature_chart_path)
    plt.close()

# Step 2: Recursive Feature Elimination (RFE)
print("\n--- Performing Recursive Feature Elimination ---")

# Get the target variable for the train set
y_merged_train = y_train

# Define the number of features to select
n_features_to_select = 50  # Reduced from 100 to 50

# Create a base estimator for RFE
base_estimator = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

# Create RFE
rfe = RFE(estimator=base_estimator, n_features_to_select=n_features_to_select, step=10)

# Convert all column names to strings before RFE
print("Converting all column names to strings...")
merged_train_final_uncorrelated.columns = merged_train_final_uncorrelated.columns.astype(str)
merged_test_final_uncorrelated.columns = merged_test_final_uncorrelated.columns.astype(str)

# Now proceed with RFE
print(f"Fitting RFE to select {n_features_to_select} features...")
rfe.fit(merged_train_final_uncorrelated, y_merged_train)

# Get selected features
selected_features = merged_train_final_uncorrelated.columns[rfe.support_]

# Create datasets with selected features
merged_train_final_selected = merged_train_final_uncorrelated[selected_features]
merged_test_final_selected = merged_test_final_uncorrelated[selected_features]

print(f"Dataset shape after RFE - train: {merged_train_final_selected.shape}")
print(f"Dataset shape after RFE - test: {merged_test_final_selected.shape}")

# Log RFE metrics with unique parameter names
if mlflow_active:
    mlflow.log_param("fs_n_features_to_select", n_features_to_select)
    mlflow.log_param("fs_features_selected_by_rfe", len(selected_features))
    mlflow.log_text("\n".join(selected_features), "selected_features.txt")
    
    # Log feature importance ranking
    feature_ranking = pd.DataFrame({
        'Feature': merged_train_final_uncorrelated.columns,
        'Ranking': rfe.ranking_
    }).sort_values('Ranking')
    
    mlflow.log_text(feature_ranking.to_string(), "feature_ranking.txt")

# After RFE feature selection
print("\n--- Features Selected by RFE ---")
print(f"Number of features selected: {len(selected_features)}")
print("Top 20 selected features:")
print("\n".join(map(str, selected_features[:20])))
if len(selected_features) > 20:
    print(f"... and {len(selected_features) - 20} more")

# Save the list of selected features to a CSV file
pd.Series(selected_features, name="selected_features").to_csv("selected_features.csv", index=False)
print("Full list of selected features saved to 'selected_features.csv'")

# Save feature ranking to a CSV file
feature_ranking.to_csv("feature_ranking.csv", index=False)
print("Feature ranking saved to 'feature_ranking.csv'")

# Final dataset for modeling
X_train_final = merged_train_final_selected
X_test_final = merged_test_final_selected
y_train_final = y_merged_train
y_test_final = y_test

print("\n--- Final Dataset Information ---")
print(f"Final train dataset shape: {X_train_final.shape}")
print(f"Final test dataset shape: {X_test_final.shape}")
print(f"Features reduced from {merged_train_final.shape[1]} to {X_train_final.shape[1]}")

# Log final dataset metrics with unique parameter names
if mlflow_active:
    mlflow.log_param("fs_final_train_rows", X_train_final.shape[0])
    mlflow.log_param("fs_final_train_cols", X_train_final.shape[1])
    mlflow.log_param("fs_final_test_rows", X_test_final.shape[0])
    mlflow.log_param("fs_final_test_cols", X_test_final.shape[1])
    mlflow.log_param("fs_total_feature_reduction", merged_train_final.shape[1] - X_train_final.shape[1])
    mlflow.log_param("fs_feature_reduction_percentage", 
                    (merged_train_final.shape[1] - X_train_final.shape[1]) / merged_train_final.shape[1] * 100)
    
    # Log final datasets
    mlflow.log_text(X_train_final.head().to_string(), "final_train_sample.txt")
    mlflow.log_text(X_train_final.describe().to_string(), "final_train_stats.txt")

# End the MLflow run for feature selection
if mlflow_active:
    mlflow.end_run()
    print("MLflow run for feature selection completed and artifacts logged.")

print("\n--- Feature Selection completed successfully ---")
print(f"Final dataset has {X_train_final.shape[1]} features after preprocessing and feature selection.")

# After RFE feature selection, add this code to analyze the source of selected features in more detail
print("\n--- Detailed Analysis of Selected Features ---")

# Count how many selected features came from identity dataset
selected_identity_features = [feat for feat in selected_features if feat in identity_columns]
selected_transaction_features = [feat for feat in selected_features if feat not in identity_columns]

print(f"Selected features from identity dataset: {len(selected_identity_features)} ({len(selected_identity_features)/len(selected_features)*100:.1f}%)")
print(f"Selected features from transaction dataset: {len(selected_transaction_features)} ({len(selected_transaction_features)/len(selected_features)*100:.1f}%)")

# Check if identity flag was selected
if identity_flag_col in selected_features:
    print(f"The identity flag '{identity_flag_col}' was selected as an important feature.")
else:
    print(f"The identity flag '{identity_flag_col}' was NOT selected as an important feature.")

# Print all identity features if any were selected
if selected_identity_features:
    print("\nAll identity features selected:")
    for i, feat in enumerate(selected_identity_features, 1):
        print(f"{i}. {feat}")

# Print top 20 transaction features if any were selected
if selected_transaction_features:
    print("\nTop 20 transaction features selected:")
    for i, feat in enumerate(selected_transaction_features[:20], 1):
        print(f"{i}. {feat}")
    if len(selected_transaction_features) > 20:
        print(f"... and {len(selected_transaction_features) - 20} more")

# Log detailed feature selection analysis to MLflow
if mlflow_active:
    # Start a new run if needed
    if not mlflow.active_run():
        mlflow.start_run(run_name=f"feature_analysis_{time.strftime('%Y%m%d_%H%M%S')}")
    
    # Log counts and percentages
    mlflow.log_param("identity_features_selected_count", len(selected_identity_features))
    mlflow.log_param("transaction_features_selected_count", len(selected_transaction_features))
    mlflow.log_metric("identity_features_selected_pct", len(selected_identity_features)/len(selected_features)*100)
    mlflow.log_metric("transaction_features_selected_pct", len(selected_transaction_features)/len(selected_features)*100)
    
    # Log identity flag selection
    mlflow.log_param("identity_flag_selected", identity_flag_col in selected_features)
    
    # Log lists of selected features by source
    mlflow.log_text("\n".join(selected_identity_features), "selected_identity_features.txt")
    mlflow.log_text("\n".join(selected_transaction_features), "selected_transaction_features.txt")
    
    # Create and log a pie chart of feature sources
    plt.figure(figsize=(8, 8))
    plt.pie([len(selected_identity_features), len(selected_transaction_features)], 
            labels=['Identity Features', 'Transaction Features'],
            autopct='%1.1f%%',
            colors=['#ff9999','#66b3ff'],
            explode=(0.1, 0))
    plt.title('Sources of Selected Features')
    plt.tight_layout()
    
    # Save and log the pie chart
    feature_source_chart_path = "feature_source_distribution.png"
    plt.savefig(feature_source_chart_path)
    mlflow.log_artifact(feature_source_chart_path)
    plt.close()
    
    # End the run if we started one
    if mlflow.active_run() and mlflow.active_run().info.run_name.startswith("feature_analysis"):
        mlflow.end_run()

print("\n--- Feature Analysis completed successfully ---")

# At the end of your preprocessing pipeline, after feature selection
print("\n--- Saving Preprocessed Data to CSV Files ---")
X_train_final.to_csv('X_train_final.csv', index=False)
X_test_final.to_csv('X_test_final.csv', index=False)
pd.DataFrame(y_train_final, columns=['target']).to_csv('y_train_final.csv', index=False)
pd.DataFrame(y_test_final, columns=['target']).to_csv('y_test_final.csv', index=False)
print("Preprocessed data saved to CSV files.") 