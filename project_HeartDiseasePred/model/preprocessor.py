import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from catboost import CatBoostClassifier
import pickle

class Preprocessor:
    _SELECTED_COLUMNS = [
        'age',
        'cholesterol',
        'heart_rate',
        'diabetes',
        # 'family_history',
        # 'smoking',
        # 'obesity',
        'alcohol_consumption',
        'exercise_hours_per_week',
        'diet',
        # 'previous_heart_problems',
        # 'medication_use',
        'stress_level',
        'sedentary_hours_per_day',
        'income',
        'bmi',
        'triglycerides',
        'physical_activity_days_per_week',
        'sleep_hours_per_day',
        # 'heart_attack_risk_(binary)',
        # 'blood_sugar',
        # 'ck-mb',
        'troponin',
        'gender',
        'systolic_blood_pressure',
        # 'diastolic_blood_pressure',
        # 'id'
    ]

    _FLOAT_TO_INT = ['physical_activity_days_per_week',
        'stress_level',
        'alcohol_consumption',
        'diabetes',
        'diet'
    ]

    _NUM_COLUMNS = ['cholesterol',
        'physical_activity_days_per_week',
        'stress_level',
        'sleep_hours_per_day',
        'bmi',
        'triglycerides',
        'heart_rate',
        'troponin',
        'sedentary_hours_per_day',
        'exercise_hours_per_week',
        'systolic_blood_pressure',
        'age',
        'income'
    ]

    def __init__(self, file_path, file_extension):
        self.file_path = file_path
        self.file_extension = file_extension

        if os.path.splitext(self.file_path)[1] == '.csv':
            self.data = pd.read_csv(file_path)
            self.status = 'ok'
        elif os.path.splitext(self.file_path)[1] == '.json':
            self.data = pd.read_json(file_path)
            self.status = 'ok'
        else:
            self.status = 'fail'

        if self.status == 'ok':
            with open('saved_model/preprocessing.pkl', 'rb') as f:
                self._data_transformer = pickle.load(f)
            self._process()


    def _process(self):
        try:
            self.data.columns = self.data.columns.str.lower().str.replace(' ', '_')
            self.id = self.data['id']
            self.data['gender'] = self.data['gender'].apply(self._fill_gender)
            self.data = self.data.loc[:, self._SELECTED_COLUMNS]
            self.data = pd.DataFrame(data=self._data_transformer.transform(self.data),
                                    columns=self._data_transformer.get_feature_names_out())
            self.data.columns = self.data.columns.str.replace('num__', '').str.replace('cat__', '')
            self.data[self._FLOAT_TO_INT] = self.data[self._FLOAT_TO_INT].astype('int')
            self.data[self._NUM_COLUMNS] = self.data[self._NUM_COLUMNS].astype('float')
            self.data = self.data[self._SELECTED_COLUMNS]
        except:
            self.status = 'fail'


    def save_data(self):
        if self.file_extension == '.csv':
            self.data.to_csv(os.path.dirname(self.file_path) + '/processed_data.csv', index=False)
        else:
            self.data.to_json(os.path.dirname(self.file_path) + '/processed_data.json', orient="records")

    @staticmethod
    def _fill_gender(x):
        if x == 'Male' or x == 'Female':
            return x
        if x == 1:
            return 'Male'
        else:
            return 'Female'