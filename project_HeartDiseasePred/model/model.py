import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from catboost import CatBoostClassifier
import pickle

class Model:

    def __init__(self,
                 data: pd.DataFrame,
                 id: pd.Series,
                 threshold: float,
                 file_path: str,
                 file_extension: str):
        self.data = data
        self.id = id
        self.threshold = threshold
        self.file_path = file_path
        self.file_extension = file_extension
        with open('saved_model/model.pkl', 'rb') as f:
            self._model = pickle.load(f)


    def predict_proba(self):
        return self._model.predict_proba(self.data)

    def predict(self):
        pred = np.vectorize(lambda x: 1 if x >= self.threshold else 0)
        predictions = pred(self.predict_proba()[:, 1])
        return predictions

    def save_predictions(self):
        df = pd.DataFrame()
        df['id'] = self.id
        df['prediction'] = list(self.predict())
        if self.file_extension == '.csv':
            df.to_csv(os.path.dirname(self.file_path) + '/predictions.csv', index=False)
        else:
            df.to_json(os.path.dirname(self.file_path) + '/predictions.json', orient="records")