import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler, OneHotEncoder, LabelEncoder

def preprocess_data(df, target_column):
    """Preprocess the data: handle missing values, normalize/standardize, and encode categorical columns."""

    # Normalize/standardize the target column based on skewness
    skewness = df[target_column].skew()
    if abs(skewness) > 1:  # Highly skewed
        scaler = MinMaxScaler()
        df[target_column + '_transformed'] = scaler.fit_transform(df[[target_column]])
        print("Applied Normalization (Min-Max Scaling)")
    else:  # Approximately symmetric
        scaler = StandardScaler()
        df[target_column + '_transformed'] = scaler.fit_transform(df[[target_column]])
        print("Applied Standardization (Z-score Scaling)")
    print("\nDataset after scaling:")
    print(df.head())

    # Encode categorical columns
    categorical_columns = df.select_dtypes(include=['object']).columns
    for column in categorical_columns:
        unique_values = df[column].unique()
        if len(unique_values) == 1:
            df.drop(columns=[column], axis=1, inplace=True)  # Drop columns with only one unique value
        elif len(unique_values) < 10:
            # One-Hot Encoding for columns with few unique values
            one_hot_encoder = OneHotEncoder(sparse_output=False, drop='first')
            encoded_data = one_hot_encoder.fit_transform(df[[column]])
            encoded_df = pd.DataFrame(encoded_data, columns=one_hot_encoder.get_feature_names_out([column]))
            df = pd.concat([df, encoded_df], axis=1)
            df.drop(columns=[column], axis=1, inplace=True)
        else:  # Label Encoding for columns with many unique values
            label_encoder = LabelEncoder()
            df[column + '_encoded'] = label_encoder.fit_transform(df[column])
            df.drop(columns=[column], axis=1, inplace=True)

    print("Data preprocessing completed.")
    #print(df.head())
    return df