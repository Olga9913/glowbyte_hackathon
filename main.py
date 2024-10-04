import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
import catboost
import optuna
from optuna.visualization import plot_optimization_history
from temperature import TemperatureCounting
from data import DataReader
from bert import BERTTransform
import argparse

def catboost_model(train_file, csv_file, output_path):
    temperature = TemperatureCounting()
    data_reader = DataReader()
    bert_vector = BERTTransform()
    
    train_data = data_reader.read(train_file)
    train_data = train_data.dropna()
    train_data = temperature.transform(train_data)
    train_data = data_reader.transform(train_data)
    train_data = bert_vector.transform(train_data)
    print("Train data is processed")

    test_data = data_reader.read(csv_file)
    test_data = test_data.dropna()
    output = pd.DataFrame(test_data['date'])
    test_data = temperature.transform(test_data)
    test_data = data_reader.transform(test_data)
    test_data = bert_vector.transform(test_data)
    print("Test data is processed")
    
    x, y = train_data.drop(columns=['target']), train_data[['target']]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.4)

    def objective(trial):
        param = {
            "objective": "MAE",
            "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
            "depth": trial.suggest_int("depth", 1, 12),
            "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
            "bootstrap_type": trial.suggest_categorical(
                "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
            ),
            "used_ram_limit": "3gb",
            "embedding_features": [3, 4]
        }

        if param["bootstrap_type"] == "Bayesian":
            param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
        elif param["bootstrap_type"] == "Bernoulli":
            param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

        gbm = catboost.CatBoostRegressor(**param)

        gbm.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=0)

        y_pred = gbm.predict(x_val)
        score = mean_absolute_error(y_val, y_pred)
        return score
    
    optuna.logging.set_verbosity(0)
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=300, n_jobs=-1, show_progress_bar=True)
    plot_optimization_history(study)

    gbm = catboost.CatBoostRegressor(**study.best_params, embedding_features=[3, 4])
    gbm.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=0)

    y_pred = gbm.predict(x_test)
    print('=========== Scores on test: =============')

    print(f'MAE: {mean_absolute_error(y_test, y_pred)}')
    print(f'MAPE: {mean_absolute_percentage_error(y_test, y_pred)}')
    print(f'R2-score: {r2_score(y_test, y_pred)}')
    
    if 'target' in test_data:
        test_data = test_data.drop(['target'], axis=1)

    y_pred = gbm.predict(test_data)
    output['pred'] = y_pred
    output.to_csv(output_path, index=False)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--train', default='train_dataset.csv', help='path to train data')
    parser.add_argument('-d', '--data', default='test_dataset.csv', help='path to test data')
    parser.add_argument('-o', '--output', default='prediction.csv', help='path to output')
    args = parser.parse_args()
    return args.train, args.data, args.output

def main(csv_file='test_dataset.csv'):
    train_path, from_path, to_path = get_args()
    catboost_model(train_path, from_path, to_path)

if __name__ == '__main__':
    main()