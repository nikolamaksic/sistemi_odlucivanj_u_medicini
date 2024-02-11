import pandas as pd
from sklearn.model_selection import train_test_split

class DataSet():
    def __init__(self, data_path: str):
        self.data = None
        self.test_size = 0.2
        self.test_valid_ratio = 0.5
        self.prepare_data(data_path)

    def prepare_data(self, data_path: str) -> None:
        if not self.data:
            self.data = self.load_data_from_csv(data_path)
        self.split_data_into_train_test_and_validate_sets()

    def split_data_into_train_test_and_validate_sets(self) -> None:
        X = self.data.drop(columns=['column_name'])
        y = self.data['column_name']
        X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=self.test_size)
        X_test, X_valid, y_test, y_valid = train_test_split(X_temp, y_temp, test_size=self.test_valid_ratio)
        self.X_train, self.X_test, self.X_valid = X_train, X_test, X_valid
        self.y_train, self.y_test, self.y_valid = y_train, y_test, y_valid

    def load_data_from_csv(self, csv_data_path):
        return pd.read_csv(csv_data_path)
