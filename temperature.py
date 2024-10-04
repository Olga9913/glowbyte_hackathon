import pandas as pd
from pandas import DataFrame
import numpy as np

class TemperatureCounting():
    def __init__(self):
        # Functions to apply to original data
        self.mutations = [
            self._temp_diff,
            self._temp_avg
        ]
    
    def _temp_diff(self, df: pd.DataFrame) -> DataFrame:
        df['temp_diff'] = df['temp_pred'] - df['temp']
        return df
    
    def _temp_avg(self, df: pd.DataFrame) -> DataFrame:
        df['temp_avg'] = df.groupby('time')['temp'].transform(np.mean)
        return df
    
    def read(self, path: str) -> DataFrame:
        return pd.read_csv(path)

    def transform(self, df: pd.DataFrame) -> DataFrame:
        for mutation in self.mutations:
            df = mutation(df)
        return df

    def read_transform(self, path: str) -> DataFrame:
        df = self.read(path)
        return self.transform(df)