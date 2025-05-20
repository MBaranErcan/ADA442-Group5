import streamlit as st

st.title("ADA442")
st.write(
"""
### Group 5

* Doğa Kadirdağ
* Furkan Ünsal
* Mustafa Baran Ercan
* Nazli İrem Akyol"""
)



import pandas as pd
from sklearn.preprocessing import MinMaxScaler


df = pd.read_csv('bank-additional.csv', sep=';', quotechar='"')

st.write(f"Rows: {df.shape[0]}")
st.write(f"Columns: {df.shape[1]}")

st.write("""
### These are the steps we are going to follow:
 1. Data cleaning: Perform necessary data cleaning operations to make sure
 the data is in a suitable format for analysis.
 2. Data preprocessing: Perform necessary data preprocessing operations such
 as feature scaling, encoding categorical variables, etc.
 3. Feature selection: Use feature selection techniques to select the most
 relevant features for the model.
 4. Model selection: Compare the performance of at least three different
 models (e.g., logistic regression, random forest, neural network) and choose
 the best one based on evaluation metrics.
 5. Hyperparameter tuning: Tune the hyperparameters of the selected model
 to improve its performance.
 6. Evaluation: Evaluate the performance of the final model using appropriate
 evaluation metrics.
 7. Deployment: Deploy the final model using streamlit and create a web
 interface for the model.


---
## 1- Data Cleaning
Our Data Cleaning involves 3 steps:

1.1- Cleaning Empty Data

1.2- Cleaning Wrong Format

1.3- Removing Duplicates

#### 1.1 - "Cleaning Empty Data"
         """
         )

         
st.write("Then let's see how many missing values we have in our dataset.")
st.write(f"Missing values: {df.isnull().sum().sum()}")
st.write("As we can see, se do not have any missing values in our dataset. Instead, we have \"unknown\" values. Let's see how many of them we have.")
st.write(f"Unknown values: {df.isin(['unknown']).sum().sum()}")

st.write("Number of rows with unknown values:")
st.write(f"Rows with unknown values: {df.isin(['unknown']).any(axis=1).sum()}")
st.write("Percentage of rows with unknown values:")
st.write(f"Percentage of rows with unknown values: {df.isin(['unknown']).any(axis=1).sum() / df.shape[0] * 100:.2f}%")

st.write("""
Since we have too many rows with at least one "unknown" value, simply dropping 1/4th of the dataset is not a good idea at all.
Yet, we can follow 3 paths:
1. Drop the rows with "unknown" values.
2. Treat "unknown" as a valid category.
3. Replace the "unknown" values with the most frequent value (mode) in that column.

We will follow all three paths and see how they affect our results.7
1. Drop the rows with "unknown" values.""")

df_unknown_dropped = df.copy()
df_unknown_dropped = df[~df.isin(['unknown']).any(axis=1)]

st.write(" 2. Treat \"unknown\" as a valid category.")
df_unknown_is_category = df.copy() # Leave as it is

st.write("3. Replace the \"unknown\" values with the most frequent value (mode) in that column.")
df_unknown_replaced = df.copy()
for column in df_unknown_replaced.columns:
    if df_unknown_replaced[column].dtype == 'object':
        mode = df_unknown_replaced[column].mode()[0]
        df_unknown_replaced[column] = df_unknown_replaced[column].replace('unknown', mode)

st.write(f"ORIGINAL DATA: 4th row, columns 6 and 7:\n{df.iloc[3, 5:7]}")
st.write("Let's see if we successfully replaced the \"unknown\" values with the mode in that column.")
st.write(f"\nNEW DATA: 4th Row, 6th and 7th columns after replacing unknown values with mode:\n{df_unknown_replaced.iloc[3, 5:7]}")


st.write("""
#### 1.2- Cleaning Wrong Format

Here are the rules defined for each column:

  ##### bank client data:

   1- age (numeric)

   2- job : type of job (categorical: "admin.","blue-collar","entrepreneur","housemaid","management","retired","self-employed","services","student","technician","unemployed","unknown")
   
   3 - marital : marital status (categorical: "divorced","married","single","unknown"; note: "divorced" means divorced or widowed)
   
   4 - education (categorical: "basic.4y","basic.6y","basic.9y","high.school","illiterate","professional.course","university.degree","unknown")
   
   5 - default: has credit in default? (categorical: "no","yes","unknown")
   
   6 - housing: has housing loan? (categorical: "no","yes","unknown")
   
   7 - loan: has personal loan? (categorical: "no","yes","unknown")
   
   ##### related with the last contact of the current campaign:
   
   8 - contact: contact communication type (categorical: "cellular","telephone") 
   
   9 - month: last contact month of year (categorical: "jan", "feb", "mar", ..., "nov", "dec")
  
  10 - day_of_week: last contact day of the week (categorical: "mon","tue","wed","thu","fri")
  
  11 - duration: last contact duration, in seconds (numeric). Important note:  this attribute highly affects the output target (e.g., if duration=0 then y="no"). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should only be included for benchmark purposes and should be discarded if the intention is to have a realistic predictive model.
  
  ##### other attributes:
  
  12 - campaign: number of contacts performed during this campaign and for this client (numeric, includes last contact)
  
  13 - pdays: number of days that passed by after the client was last contacted from a previous campaign (numeric; 999 means client was not previously contacted)
  
  14 - previous: number of contacts performed before this campaign and for this client (numeric)
  
  15 - poutcome: outcome of the previous marketing campaign (categorical: "failure","nonexistent","success")
  
   ##### social and economic context attributes
  
  16 - emp.var.rate: employment variation rate - quarterly indicator (numeric)
  
  17 - cons.price.idx: consumer price index - monthly indicator (numeric)     
  
  18 - cons.conf.idx: consumer confidence index - monthly indicator (numeric)     
  
  19 - euribor3m: euribor 3 month rate - daily indicator (numeric)
  
  20 - nr.employed: number of employees - quarterly indicator (numeric)

Result:

21 - y: 0, or 1

#### Remove unexpected values for each categorical column"""
)

valid_categories = {
    'job': {"admin.", "blue-collar", "entrepreneur", "housemaid", "management", 
            "retired", "self-employed", "services", "student", "technician", "unemployed", "unknown"},
    'marital': {"married", "single", "divorced", "unknown"},
    'education': {"basic.4y", "basic.6y", "basic.9y", "high.school", "illiterate", 
                  "professional.course", "university.degree", "unknown"},
    'default': {"no", "unknown", "yes"},
    'housing': {"no", "unknown", "yes"},
    'loan': {"no", "unknown", "yes"},
    'contact': {"cellular", "telephone"},
    'month': {"jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"},
    'day_of_week': {"mon", "tue", "wed", "thu", "fri"},
    'poutcome': {"failure", "nonexistent", "success"},
    'y': {"no", "yes"}
}

for col, valid_vals in valid_categories.items():
    found_invalid = False
    unique_vals = set(df[col].unique())
    invalid_vals = unique_vals - valid_vals
    if invalid_vals:
        st.write(f"Column '{col}' has invalid values: {invalid_vals}")
        found_invalid = True
if not found_invalid:
    st.write("No unexpected values found in any column.")

    
st.write("#### 1.3- Removing duplicates")
st.write("As you can see from the below, we do not have duplicate data")
duplicates = df[df.duplicated()]
st.write(duplicates)



st.write("""
## 2- Data preprocessing

In this step, we will aim to convert our dataframe into a format that is suitable for machine learning algorithms.

Data processing steps:
1. Drop the unnecessary columns
2. Change the "pdays" column
3. Apply one-hot encoding to the categorical columns
4. Apply min-max scaling to the numerical columns
5. Capping the outliers
6. Split the dataset into training and testing sets

### 2.1. Drop the unnecessary columns

Columns discussed for dropping:

1- Contact: Contact communication type (categorical: "cellular","telephone"). At first, we considered dropping the "contact" column, thinking it might not significantly affect the outcome. However, after analyzing the data, we noticed a meaningful difference in success rates: only about 5.2% of clients contacted via telephone subscribed to a term deposit, while 14.1% of those contacted via cellular did. This distinction showed us that the contact method could influence client behavior, so we decided to keep this column.

2- Duration: Last contact duration, in seconds. This attribute highly affects the output target (e.g., if duration=0 then y="no"). Yet, the duration is not known before a call is performed. Also, after the end of the call y is obviously known. Thus, this input should be discarded if the intention is to have a realistic predictive model.
""")

#drop contact column
#df_unknown_dropped = df_unknown_dropped.drop(columns=['contact'])
#df_unknown_is_category = df_unknown_is_category.drop(columns=['contact'])
#df_unknown_replaced = df_unknown_replaced.drop(columns=['contact'])

#drop duration
df_unknown_dropped = df_unknown_dropped.drop(columns=['duration'])
df_unknown_is_category = df_unknown_is_category.drop(columns=['duration'])
df_unknown_replaced = df_unknown_replaced.drop(columns=['duration'])

# print new dataframes' shape
st.write(f"DF unknown dropped: {df_unknown_dropped.shape}")
st.write(f"DF unknown is category: {df_unknown_is_category.shape}")
st.write(f"DF unknown replaced: {df_unknown_replaced.shape}")

st.write("### 2.2- Change the \"pdays\" column")
def map_pdays(val):
    if val == 999:
        return "never_contacted"
    elif 0 <= val <= 6:
        return "this_week"
    elif 7 <= val <= 13:
        return "last_week"
    elif 14 <= val < 999:
        return "older"
    else:
        st.write(f"Unexpected value: {val}")
        return val 

df_unknown_dropped['pdays'] = df_unknown_dropped['pdays'].apply(map_pdays)
df_unknown_is_category['pdays'] = df_unknown_is_category['pdays'].apply(map_pdays)
df_unknown_replaced['pdays'] = df_unknown_replaced['pdays'].apply(map_pdays)
st.write(f"DF unknown dropped:\n{df_unknown_dropped['pdays'].value_counts()}")
st.write(f"DF unknown is category:\n{df_unknown_is_category['pdays'].value_counts()}")
st.write(f"DF unknown replaced:\n{df_unknown_replaced['pdays'].value_counts()}")


st.write(" ### 2.3- Apply One-Hot Encoding to Categorical Columns")
categorical_columns = [
    "job", "marital", "education", "default", "contact",
    "housing", "loan", "month", "day_of_week",
    "poutcome", "pdays"
]
df_unknown_dropped      = pd.get_dummies(df_unknown_dropped, columns=categorical_columns, drop_first=False)
df_unknown_is_category  = pd.get_dummies(df_unknown_is_category, columns=categorical_columns, drop_first=False)
df_unknown_replaced     = pd.get_dummies(df_unknown_replaced, columns=categorical_columns, drop_first=False)

st.write(f"df_unknown_dropped new shape after encoding: {df_unknown_dropped.shape}")
st.write(f"df_unknown_is_category new shape after encoding: {df_unknown_is_category.shape}")
st.write(f"df_unknown_replaced new shape after encoding: {df_unknown_replaced.shape}")

st.write("### 2.4-Apply min-max scaling to the numerical columns")

numerical_cols = [
    'age', 'campaign', 'previous',
    'emp.var.rate', 'cons.price.idx', 
    'cons.conf.idx', 'euribor3m', 'nr.employed'
]

scaler = MinMaxScaler()

df_unknown_dropped[numerical_cols] = scaler.fit_transform(df_unknown_dropped[numerical_cols])
df_unknown_is_category[numerical_cols] = scaler.fit_transform(df_unknown_is_category[numerical_cols])
df_unknown_replaced[numerical_cols] = scaler.fit_transform(df_unknown_replaced[numerical_cols])

st.write("df_unknown_dropped first 5 rows:\n", df_unknown_dropped.head(), "\n")
st.write("df_unknown_is_category first 5 rows:\n", df_unknown_is_category.head(), "\n")
st.write("df_unknown_replaced first 5 rows:\n", df_unknown_replaced.head(), "\n")

st.write("### 2.5- Split the dataset into training and testing sets")
from sklearn.model_selection import train_test_split

X_dropped = df_unknown_dropped.drop(columns=['y'])
y_dropped = df_unknown_dropped['y']

X_is_category = df_unknown_is_category.drop(columns=['y'])
y_is_category = df_unknown_is_category['y']

X_replaced = df_unknown_replaced.drop(columns=['y'])
y_replaced = df_unknown_replaced['y']

X_train_dropped, X_test_dropped, y_train_dropped, y_test_dropped = train_test_split(X_dropped, y_dropped, test_size=0.2, stratify=y_dropped, random_state=10)

X_train_is_category, X_test_is_category, y_train_is_category, y_test_is_category = train_test_split(X_is_category, y_is_category, test_size=0.2, stratify=y_is_category, random_state=10)

X_train_replaced, X_test_replaced, y_train_replaced, y_test_replaced = train_test_split(X_replaced, y_replaced, test_size=0.2, stratify=y_replaced, random_state=10)
# Print the shapes of the train and test sets
st.write(f"X_train_dropped shape: {X_train_dropped.shape}")
st.write(f"X_test_dropped shape: {X_test_dropped.shape}")
st.write(f"y_train_dropped shape: {y_train_dropped.shape}")
st.write(f"y_test_dropped shape: {y_test_dropped.shape}")
st.write("-" * 50)
st.write(f"X_train_is_category shape: {X_train_is_category.shape}")
st.write(f"X_test_is_category shape: {X_test_is_category.shape}")
st.write(f"y_train_is_category shape: {y_train_is_category.shape}")
st.write(f"y_test_is_category shape: {y_test_is_category.shape}")
st.write("-" * 50)
st.write(f"X_train_replaced shape: {X_train_replaced.shape}")
st.write(f"X_test_replaced shape: {X_test_replaced.shape}")
st.write(f"y_train_replaced shape: {y_train_replaced.shape}")
st.write(f"y_test_replaced shape: {y_test_replaced.shape}")


st.write("### 3- Feature selection")
st.write("#### 3.1 - Recursive Feature Elimination (RFE)")
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier

estimator = GradientBoostingClassifier(n_estimators=100, random_state=10)
rfe = RFE(estimator, n_features_to_select=10, step=1)

# Model 1: Dropped unknown values
rfe = rfe.fit(X_train_dropped, y_train_dropped)
st.write("Selected features (Model-1):")
st.write(X_train_dropped.columns[rfe.support_])

# Model 2: Unknown values as a valid category
rfe = rfe.fit(X_train_is_category, y_train_is_category)
st.write("\nSelected features (Model-2):")
st.write(X_train_is_category.columns[rfe.support_])

# Model 3: Replaced unknown values with the mode
rfe = rfe.fit(X_train_replaced, y_train_replaced)
st.write("\nSelected features (Model-3):")
st.write(X_train_replaced.columns[rfe.support_])

st.write("### 4. Model selection")
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=10, class_weight='balanced'),
    "Decision Tree": DecisionTreeClassifier(random_state=10, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=10, class_weight='balanced'),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=10),
    "Neural Network": MLPClassifier(max_iter=1000, random_state=10)
}

def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label='yes')
    recall = recall_score(y_test, y_pred, pos_label='yes')
    f1 = f1_score(y_test, y_pred, pos_label='yes')

    return accuracy, precision, recall, f1

results = {}
for model_name, model in models.items():
    results[model_name] = {}

    # Model 1: Dropped unknown values
    accuracy, precision, recall, f1 = evaluate_model(model, X_train_dropped, y_train_dropped, X_test_dropped, y_test_dropped)
    results[model_name]['Dropped Unknown'] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

    # Model 2: Unknown values as a valid category
    accuracy, precision, recall, f1 = evaluate_model(model, X_train_is_category, y_train_is_category, X_test_is_category, y_test_is_category)
    results[model_name]['Unknown as Category'] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

    # Model 3: Replaced unknown values with the mode
    accuracy, precision, recall, f1 = evaluate_model(model, X_train_replaced, y_train_replaced, X_test_replaced, y_test_replaced)
    results[model_name]['Replaced with Mode'] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1
    }

    sorted_results = {
    model: dict(
        sorted(
            v.items(),
            key=lambda item: item[1]['F1 Score'],
            reverse=True
        )
    )
    for model, v in results.items()
}
# Print the results
st.write("\nModel Evaluation Results:")
st.write(f"{'Model':<30} {'Dataset':<30} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10}")
for model, datasets in sorted_results.items():
    for dataset, metrics in datasets.items():
        accuracy = metrics['Accuracy']
        precision = metrics['Precision']
        recall = metrics['Recall']
        f1 = metrics['F1 Score']
        st.write(f"{model:<30} {dataset:<30} {accuracy:<10.4f} {precision:<10.4f} {recall:<10.4f} {f1:<10.4f}")

st.write("### 5- Hyperparameter tuning:")


from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier

# 1) Same model set
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=10, class_weight='balanced'),
    "Decision Tree": DecisionTreeClassifier(random_state=10, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=10, class_weight='balanced'),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=10),
    "Neural Network": MLPClassifier(max_iter=1000, random_state=10)
}


# 2) Hyperparameter grids
param_grid = {
    "Logistic Regression": {
        'C': [0.01, 0.1, 1, 10],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga']
    },
    "Decision Tree": {
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    },
    "Random Forest": {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    },
    "Gradient Boosting": {
        'n_estimators': [50, 100],
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5]
    },
    "Neural Network": {
        'hidden_layer_sizes': [(100), (50, 50), (100, 100, 100)],
        'activation': ['relu'],
        'solver': ['adam', 'sgd'],
        'alpha': [0.0001, 0.001]
    }
}

# 3) Function to tune hyperparameters
def tune_hyperparameters(model, grid, X_tr, y_tr):
    grid_search = GridSearchCV(
        estimator   = model,
        param_grid  = grid,
        scoring     = 'f1_macro',       # use macro-F1 instead of binary F1
        cv          = StratifiedKFold(n_splits=3, shuffle=True, random_state=42),
        n_jobs      = -1,                
        error_score=0 # treat failed configurations as score=0
    )
    grid_search.fit(X_tr, y_tr)
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_

    
# 4) Run tuning on each dataset variant
dataframes = {
    "Dropped Unknown": (X_train_dropped, y_train_dropped),
    "Unknown as Category": (X_train_is_category, y_train_is_category),
    "Replaced with Mode": (X_train_replaced, y_train_replaced),
}

tuned_results = {}
for name, model in models.items():
    tuned_results[name] = {}
    for variant, (X_tr, y_tr) in dataframes.items():
        best_m, best_p, best_s = tune_hyperparameters(model, param_grid[name], X_tr, y_tr)
        tuned_results[name][variant] = {
            'best_model': best_m,
            'best_params': best_p,
            'best_f1_macro': best_s
        }




# 5) Display all results
st.write("Tuned Hyperparameters (scoring=f1_macro):")
st.write(f"{'Model':<30} {'Dataset':<30} {'Best F1 Macro':<15} {'Best Params':<50}")
for model, datasets in tuned_results.items():
    for dataset, metrics in datasets.items():
        best_f1_macro = metrics['best_f1_macro']
        best_params = metrics['best_params']
        st.write(f"{model:<30} {dataset:<30} {best_f1_macro:<15.4f} {str(best_params):<50}")

        
# 6) Print only the best results for each model
st.write("\nBest Hyperparameters for each model:")
for model, datasets in tuned_results.items():
    best_dataset = max(datasets.items(), key=lambda x: x[1]['best_f1_macro'])
    best_variant, metrics = best_dataset
    best_f1_macro = metrics['best_f1_macro']
    best_params = metrics['best_params']
    st.write(f"{model:<30} {best_variant:<30} {best_f1_macro:<15.4f} {str(best_params):<50}")

# Print best of best
best_model = None
best_f1_macro = 0
best_model_name = ""
best_variant = ""
for model, datasets in tuned_results.items():
    for dataset, metrics in datasets.items():
        if metrics['best_f1_macro'] > best_f1_macro:
            best_f1_macro = metrics['best_f1_macro']
            best_model = metrics['best_model']
            best_model_name = model
            best_variant = dataset


st.write(f"\nBest Model: (WINNER)")
st.write(f"{model:<30} {best_variant:<30} {best_f1_macro:<15.4f} {str(best_params):<50}")

from sklearn.pipeline import Pipeline
king_model = RandomForestClassifier(max_depth=10, min_samples_leaf=2, min_samples_split=2, n_estimators=200)

pipeline = Pipeline([
    ('scaler', MinMaxScaler()),
    ('model', best_model)
])
# Fit the pipeline on the training data
pipeline.fit(X_train_dropped, y_train_dropped)
# Make predictions on the test data
y_pred = pipeline.predict(X_test_dropped)
# Evaluate the model
accuracy = accuracy_score(y_test_dropped, y_pred)
precision = precision_score(y_test_dropped, y_pred, pos_label='yes')
recall = recall_score(y_test_dropped, y_pred, pos_label='yes')
f1 = f1_score(y_test_dropped, y_pred, pos_label='yes')
st.write(f"\nPipeline Model Evaluation:")
st.write(f"{'Model':<30} {'Dataset':<30} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1 Score':<10}")



st.markdown("---")
st.markdown("## Let's try ourself with fresh data!")


# Create input widgets
age = st.slider("Age", 18, 95, 30)
campaign = st.slider("Number of contacts during this campaign", 1, 30, 1)
previous = st.slider("Previous contacts", 0, 10, 0)
emp_var_rate = st.number_input("Employment variation rate", value=1.1)
cons_price_idx = st.number_input("Consumer price index", value=93.2)
cons_conf_idx = st.number_input("Consumer confidence index", value=-40.0)
euribor3m = st.number_input("Euribor 3-month rate", value=4.0)
nr_employed = st.number_input("Number of employees", value=5191.0)

# Create a new sample with the right column order
sample = pd.DataFrame([{
    'age': age,
    'campaign': campaign,
    'previous': previous,
    'emp.var.rate': emp_var_rate,
    'cons.price.idx': cons_price_idx,
    'cons.conf.idx': cons_conf_idx,
    'euribor3m': euribor3m,
    'nr.employed': nr_employed
}])

# Use the trained pipeline to predict
if st.button("Predict Subscription"):
    prediction = pipeline.predict(sample)[0]
    probability = pipeline.predict_proba(sample)[0][1]  # probability of 'yes'
    st.success(f"Prediction: {prediction.upper()} (Confidence: {probability:.2%})")