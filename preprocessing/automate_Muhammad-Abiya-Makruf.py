from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from joblib import dump
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import pandas as pd

def preprocess_data(data, target_column, save_path, file_path):
    numeric_features = data.select_dtypes(include=['float64','int64']).columns.tolist()
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()
    column_names = data.columns.drop(target_column)

    if target_column in numeric_features:
        numeric_features.remove(target_column)
    if target_column in categorical_features:
        categorical_features.remove(target_column)

    # Pipeline
    numeric_transformer = Pipeline(steps=[
        ('scaler', MinMaxScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('encoder', OneHotEncoder())
    ])

    preprocessor = ColumnTransformer(
        transformers = [
            ('num',numeric_transformer, numeric_features),
            ('cat',categorical_transformer, categorical_features)
        ]
    )

    # simpan target
    y = data[target_column]
    X = data.drop(columns=[target_column])

    # fit transform
    X_processed = preprocessor.fit_transform(X)

    # DataFrame hasil + target
    df_final = pd.DataFrame(
        X_processed.toarray() if hasattr(X_processed, "toarray") else X_processed
    )
    df_final[target_column] = y.reset_index(drop=True)

    # save pipeline dan csv
    dump(preprocessor, save_path)
    df_final.to_csv(file_path, index=False)

# # Contoh penggunaan
# data = pd.read_csv("../insurance_raw.csv")
# preprocess_data(data,"charges","preprocessor.joblib","insurance_preprocessing.csv")