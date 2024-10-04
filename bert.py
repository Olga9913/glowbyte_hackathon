import pandas as pd
from pandas import DataFrame
import torch
from transformers import AutoTokenizer, AutoModel

class BERT:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
    
    def data_text(self, dataframe):
        pred = set(dataframe['weather_pred'].values)
        fact = set(dataframe['weather_fact'].values)
        text_data = list(pred.union(fact))
        return text_data
    
    def bert_embeddings_dict(self, text_data):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)
        text_bert = {}
        n = len(text_data)
        for i in range(n):
            encodings = self.tokenizer(text_data[i], padding=True, return_tensors='pt', max_length=50, add_special_tokens = True)
            encodings = encodings.to(device)
            with torch.no_grad():
                embeds = self.model(**encodings)
            sentence_embedding = embeds[0][0, 0, :].cpu().tolist()
            text_bert[text_data[i]] = sentence_embedding
        return text_bert

class BERTTransform:
    def __init__(self):
        # Functions to apply to original data
        self.mutations = [
            self._extract_bert,
            self._metrics_count
        ]
    
    def _drop_na(self, df: DataFrame) -> DataFrame:
        df = df.dropna()
        return df
    
    def _extract_bert(self, df: DataFrame) -> DataFrame:
        bert_tokenizer = AutoTokenizer.from_pretrained('cointegrated/rubert-tiny')
        bert_model = AutoModel.from_pretrained('cointegrated/rubert-tiny', output_hidden_states=True)
        bert_model = bert_model.eval()
        bert = BERT(bert_model, bert_tokenizer)
        text_data = bert.data_text(df)
        bert_dict = bert.bert_embeddings_dict(text_data)
        df['weather_pred'] = df['weather_pred'].apply(lambda x: bert_dict.get(x))
        df['weather_fact'] = df['weather_fact'].apply(lambda x: bert_dict.get(x))
        return df
    
    def _metrics_count(self, df: DataFrame) -> DataFrame:
        n = df.shape[0]
        bert_1 = list(df['weather_pred'].values)
        bert_2 = list(df['weather_fact'].values)
        bert_cossim, bert_pairwise = [0] * n, [0] * n
        for i in range(n):
            bert_1[i] = torch.FloatTensor(bert_1[i]).unsqueeze(dim=0)
            bert_2[i] = torch.FloatTensor(bert_2[i]).unsqueeze(dim=0)
            bert_cossim[i] = (torch.nn.functional.cosine_similarity(bert_1[i], bert_2[i])).item()
            bert_pairwise[i] = (torch.nn.functional.pairwise_distance(bert_1[i], bert_2[i])).item()
        df['cossim'] = bert_cossim
        df['pairwise'] = bert_pairwise
        return df
    
    def _sep_vectors(self, df: DataFrame) -> DataFrame:
        w_pred_columns = [f'w_pred_{i}' for i in range(312)]
        w_fact_columns = [f'w_fact_{i}' for i in range(312)]
        df[w_pred_columns] = df['weather_pred'].values.tolist()
        df[w_fact_columns] = df['weather_fact'].values.tolist()
        return df
    
    def _tmp_drop_weather(self, df: DataFrame) -> DataFrame:
        df = df.drop(columns=['weather_pred', 'weather_fact'], axis=1)
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