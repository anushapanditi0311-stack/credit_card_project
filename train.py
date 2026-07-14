import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def run_training_pipeline():
    print("Step 1: Loading raw datasets...")
    app_df = pd.read_csv('data/application_record.csv')
    credit_df = pd.read_csv('data/credit_record.csv')

    print("Step 2: Constructing target variable and merging records...")
    # Label status '2', '3', '4', or '5' as risky (1)
    credit_df['is_risky'] = credit_df['STATUS'].isin(['2', '3', '4', '5']).astype(int)
    
    # Identify if a client has defaulted at least once
    user_target = credit_df.groupby('ID')['is_risky'].max().reset_index()
    user_target.rename(columns={'is_risky': 'target'}, inplace=True)
    
    # Merge datasets
    df = pd.merge(app_df, user_target, on='ID', how='inner')

    print("Step 3: Preprocessing and resolving anomalies...")
    # Replace null occupation entries
    df['OCCUPATION_TYPE'] = df['OCCUPATION_TYPE'].fillna('Unknown')
    
    # Convert birth and employment day offsets to years
    df['AGE'] = (df['DAYS_BIRTH'] / -365.25).astype(int)
    df['YEARS_EMPLOYED'] = df['DAYS_EMPLOYED'].apply(lambda x: 0 if x > 0 else x / -365.25)
    df['UNEMPLOYED'] = (df['DAYS_EMPLOYED'] > 0).astype(int)
    
    # Drop raw and redundant ID/offset variables
    df_clean = df.drop(columns=['ID', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'FLAG_MOBIL'])

    print("Step 4: Categorical Label Encoding...")
    le = LabelEncoder()
    cat_cols = ['CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE', 
                'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE']
    for col in cat_cols:
        df_clean[col] = le.fit_transform(df_clean[col].astype(str))

    # Separate features and target
    X = df_clean.drop(columns=['target'])
    y = df_clean['target']

    # Stratified split to keep target ratio balanced
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("\nStep 5: Training and evaluating classifiers...")
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced'),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    }

    best_acc = 0.0
    best_model = None
    best_model_name = ""

    for name, clf in models.items():
        # Fit classifier
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        print(f"-> {name} Accuracy: {acc * 100:.2f}%")
        
        # Track the highest performing model
        if acc > best_acc:
            best_acc = acc
            best_model = clf
            best_model_name = name

    print(f"\nWinner: {best_model_name} with {best_acc * 100:.2f}% accuracy.")
    
    # Print metrics report for the winner
    winner_preds = best_model.predict(X_test)
    print("\nBest Model Performance Report:")
    print(classification_report(y_test, winner_preds))

    # Save model binary file
    print("Step 6: Serializing and saving best model...")
    os.makedirs('models', exist_ok=True)
    joblib.dump(best_model, 'models/card_model.joblib')
    print("Model successfully written into models/card_model.joblib")

if __name__ == '__main__':
    run_training_pipeline()
