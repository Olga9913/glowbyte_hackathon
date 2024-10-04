import pandas as pd
from pandas import DataFrame

class DataReader():
    def __init__(self):
        # Functions to apply to original data
        self.mutations = [
            self._apply_datetime,
            self._extract_dayofweek,
            self._extract_date,
            self._extract_month,
            self._extract_quarter,
            self._extract_year,
            self._tmp_drop_weather
        ]
    
    def _drop_na(self, df: DataFrame) -> DataFrame:
        df = df.dropna()
        return df

    def _tmp_drop_weather(self, df: DataFrame) -> DataFrame:
        df = df.drop(columns=['date'])
        return df

    def _apply_datetime(self, df: DataFrame) -> DataFrame:
        df['date'] = pd.to_datetime(df['date'])
        return df

    def _extract_month(self, df: DataFrame) -> DataFrame:
        df['month'] = df['date'].dt.month
        return df

    def _extract_year(self, df: DataFrame) -> DataFrame:
        df['year'] = 2023 - df['date'].dt.year
        return df
    
    def _extract_quarter(self, df: DataFrame) -> DataFrame:
        df['quarter'] = df['date'].dt.quarter
        return df

    def _extract_date(self, df: DataFrame) -> DataFrame:
        df['day'] = df['date'].dt.day
        return df
    
    def _extract_dayofweek(self, df: DataFrame) -> DataFrame:
        df['dayofweek'] = df['date'].dt.dayofweek
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
