import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, LabelEncoder


def preprocess_data(input_path: str, output_path: str):
    """
    berfungsi untuk melakukan preprocessing data credit card secara otomatis.
    Menghasilkan dataset yang siap digunakan untuk training model.
    """

    #melakukan load pada dataset
    df = pd.read_csv(input_path)

    #menghapus duplikasi (jika ada)
    df = df.drop_duplicates()

    #melakukan pemisahan fitur dan target
    X = df.drop("default.payment.next.month", axis=1)
    y = df["default.payment.next.month"]

    #melakukan encoding
    categorical_cols = ["SEX", "EDUCATION", "MARRIAGE"]
    le = LabelEncoder()
    for col in categorical_cols:
        X[col] = le.fit_transform(X[col])

    #melakukan scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #melakukan penggabungan kembali fitur dan target
    df_preprocessed = pd.DataFrame(X_scaled, columns=X.columns)
    df_preprocessed["target"] = y.values

    #menyimpan hasil dari preprocessing yang dilakukan
    df_preprocessed.to_csv(output_path, index=False)

    return df_preprocessed


if __name__ == "__main__":
    input_file = "../UCI_Credit_Card.csv"
    output_file = "credit_card_preprocessing.csv"

    preprocess_data(input_file, output_file)