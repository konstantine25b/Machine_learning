# competition House Prices - Linear Regresion

## Setup and Execution

1.  **Clone the Repository (if applicable):**
    ```bash
    git clone https://github.com/konstantine25b/Machine_learning
    cd assignment1
    ```
2.  **Install Dependencies:** It is highly recommended to use a virtual environment.

    * **Create a virtual environment (if you haven't already):**
        ```bash
        python -m venv venv  # For Python 3.3+
        # OR
        virtualenv venv       # If you have virtualenv installed
        ```

    * **Activate the virtual environment:**
        ```bash
        source venv/bin/activate  # On macOS/Linux
        venv\Scripts\activate  # On Windows
        ```

    * **Install the required packages from `requirements.txt`:**
        ```bash
        pip install -r requirements.txt
        ```

3.  **Run the Script:** Execute the main Python script of the project.


ესე გადმოიწერ და დააყენებ ლოკალურად.

ახლა თვითონ დავალებაზე გადავიდეთ:

პირველ რიგში გვაქვს modek_experiment.ipynb ფაილი სადაც ხდება მოდელზე ყველაფერი დავიწყოთ ამ ფაილის გარჩევა:

# Distribution of the Target Variable
print("Plotting distribution of the target variable (SalePrice)...")
plt.figure(figsize=(10, 6))
sns.histplot(y_train, kde=True, bins=50)
plt.title("Distribution of the Target Variable (SalePrice) (Training Set)")
plt.xlabel("SalePrice")
plt.ylabel("Frequency")
plt.show()

ამ ნახაზით გავიგე როგორი განაწილება ჰქონდა მინაცემებს და დავიწყე დათას გაწმენდა.

z_score = np.abs(stats.zscore(y_train))
outlier_indices = np.where(z_score > OUTLIER_Z_THRESHOLD)[0]
outlier_original_indices = y_train.iloc[outlier_indices].index # Get original df indices
num_outliers = len(outlier_indices)

ასე აღმოვაჩინე აუთლაიერები და წავშალე დათადან.

 if 'YrSold' in df_eng.columns and 'YearRemodAdd' in df_eng.columns:
        df_eng['TotalRemodYears'] = df_eng['YrSold'] - df_eng['YearRemodAdd']
    if 'YrSold' in df_eng.columns and 'YearBuilt' in df_eng.columns:
        df_eng['TotalBuiltAge'] = df_eng['YrSold'] - df_eng['YearBuilt']
    if 'GrLivArea' in df_eng.columns and 'LotArea' in df_eng.columns:
        df_eng['LivingAreaRatio'] = df_eng['GrLivArea'] / (df_eng['LotArea'] + 1e-6)
    if 'TotalBsmtSF' in df_eng.columns and 'LotArea' in df_eng.columns:
        df_eng['BasementSurfaceRatio'] = df_eng['TotalBsmtSF'] / (df_eng['LotArea'] + 1e-6)
    if 'GarageArea' in df_eng.columns:
        df_eng['HasGarage'] = (df_eng['GarageArea'] > 0).astype(int)
    if 'TotalBsmtSF' in df_eng.columns:
        df_eng['HasBasement'] = (df_eng['TotalBsmtSF'] > 0).astype(int)
დავამატე რამდენიმე სვეტი (ფიჩერი) რომლებიც ასახავდა სხვადასხვა რიცხვულლ, პროცენტულ, და ბაინერი( თრუ ან ფოლს) დატას.


class HighCorrelationRemover(BaseEstimator, TransformerMixin): მაქბს კლასი რომელსაც ფაიფლაინში ვიყენებ მაღალ კორელირებული ცვლადების ამოსაგდებად ამ შემთხვევაში >0.7

# Define numerical pipeline steps
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
]) რიცხვობრივ დათაში ნალებს ვანაცვლებ მედიანით, ხოლო

# Define categorical pipeline steps
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False)) # sparse=False for easier handling downstream
]) კატეგორიულში ყველაზე ხშირით. გვაქვს OneHotEncoding- რაც კატეგორიულ სვეტებს გარდაქმნის რამდენიმე ბაინერი სვეტად.

ამის მერე შევქმენი გფაიფლაინი სადსაც ამ პრე პროცესინგს ვაერთიანებ.
ასევე ვfit-ავ ჩვენ დატას ამ პრე პროცესებზე.

შემდეგ გვაქ RFE - ანუ ვშლით ნაკლებად საჭირო სვეტებს და ვტოვებთ ყველაზე საჭირო 25 სვეტს.

ამის მერე გვაქ უკვე ტრენინგი და Kfold-ის საუალებით 5 ჯერ ვსფლიტავთ და gridSearch-ს ვიყენებთ საუკეთესო ალგორითმის საპოვნელად .

შეფასების მეტრიკად ვიყენებთ : scoring = ['neg_root_mean_squared_error', 'neg_mean_absolute_error', 'r2']

რაც შეეხება ალგორითმებს გვაქ ვს 2: LinearRegression და DecisionTreeRegressor. მაგრამ ასევევ გვაქვს Ridge რომელიც ლინეარ რეგრესიის რეგულარიზებული ვარიანტია.

ვატრეინინგებთ სხვადასხვა ჰიპერპარამეტრებზე. ასევე სტანდარტიზაციისთვის გვაქ ორი მიდგომა: StandardScaler() და MinMaxScaler()

ამის მერე იწყება ტრეინინგი:
შედეგებში საუკეთესი აღმოჩნდა წრფივი რეგრესია StandardScaler()-ით:  https://dagshub.com/konstantine25b/Machine_learning.mlflow/#/experiments/0/runs/07ab81f377e64097a7cc0a0ef64986c8

Best CV Results for LinearRegression (RFE: StandardScaler):
  Best CV RMSE: 29058.2500
  Best CV MAE:  15618.0925
  Best CV R2:   0.7932
  Best Params:  {}

Training Set Performance (LinearRegression, RFE: StandardScaler):
  Training RMSE: 20974.3336
  Training MAE:  14329.9959
  Training R2:   0.9015

Validation Set Performance (LinearRegression, RFE: StandardScaler):
  Validation RMSE: 30185.0784
  Validation MAE:  17602.0426
  Validation R2:   0.8812


მიუხედავად ამისა ვხედავთ რომ გვაქვს დიდი ვარიაცია RMSE რადგან ტრაინინგსა და ტესტსეტს შორის განსხვავება საშუალოდ 10000მდეა,  MAE - ამაში ვარიაცია დაბალი გვაქ , R2 - ეს მაღალი მაჩვენებელია ანუ ნაკლები გვაქ bias. 

DecisionTree-ებში ,  - ში  ოვერფიტი გვაქ რადგან, საკამოდ დიდია განსხვავება ტრეინსეტსა და ვალიდაციის სეტს შორის.
LinearRegression- minMax- ამან მსგავსი შედეგი დადო რაც წრფივი რეგრესიი სტანდარტ სქეილერმა მაგრამ ოდნავ უარესი

Ringe- ებმა ორივემ უფრო ცუდი შედეგები დადეს ვიდრე უბრალო წრფივმა რეგრესიებმა.

შესაბამისად ამ 6 მოდელიდან LinearRegression, RFE: StandardScaler შედარებით ყველაზე კარგად იმუშავა
 
