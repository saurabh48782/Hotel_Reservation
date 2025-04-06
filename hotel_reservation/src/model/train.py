import os
import sys

import dvc.api
import joblib
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
import mlflow
import numpy as np
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit

from dvclive import Live
from hotel_reservation.src.logger import get_logger
from hotel_reservation.utils import read_csv_files
from hotel_reservation.utils.custom_exception import CustomException

logger = get_logger(__name__)

params = dvc.api.params_show()


def get_best_hyperparams(X_train, y_train):
    try:
        logger.info(">>>>>>>> Hyper-parametertuning started")

        tscv = TimeSeriesSplit(n_splits=5)
        study = optuna.create_study(direction="maximize")

        def objective(trial):
            learning_rate = trial.suggest_float("learning_rate", 0.001, 0.3)
            num_leaves = trial.suggest_int("num_leaves", 2, 256)
            max_depth = trial.suggest_int("max_depth", -1, 20)
            min_child_samples = trial.suggest_int("min_child_samples", 5, 100)
            subsample = trial.suggest_float("subsample", 0.5, 1.0)
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5,
                                                   1.0)
            n_estimators = trial.suggest_int("n_estimators", 100, 5000)
            reg_alpha = trial.suggest_float("reg_alpha", 1e-5, 10.0, log=True)
            reg_lambda = trial.suggest_float("reg_lambda",
                                             1e-5,
                                             10.0,
                                             log=True)

            auc_scores = []

            for train_idx, val_idx in tscv.split(X_train):
                X_t, X_v = X_train.iloc[train_idx], X_train.iloc[val_idx]
                y_t, y_v = y_train.iloc[train_idx], y_train.iloc[val_idx]
                model = LGBMClassifier(
                    learning_rate=learning_rate,
                    num_leaves=num_leaves,
                    max_depth=max_depth,
                    min_child_samples=min_child_samples,
                    subsample=subsample,
                    colsample_bytree=colsample_bytree,
                    n_estimators=n_estimators,
                    reg_alpha=reg_alpha,
                    reg_lambda=reg_lambda,
                    class_weight='balanced',
                    random_state=params['model_features']['random_state'],
                    n_jobs=-1,
                    verbosity=-1,
                )
                model.fit(
                    X_t,
                    y_t,
                    eval_set=[(X_v, y_v)],
                    eval_metric=params['model_features']['eval_metric'],
                    callbacks=[
                        early_stopping(stopping_rounds=params['model_features']
                                       ['stopping_rounds']),
                        log_evaluation(period=1000)
                    ])

                y_pred = model.predict_proba(X_v)[:, 1]
                auc_scores.append(roc_auc_score(y_v, y_pred))

            return np.mean(auc_scores)

        study.optimize(objective, n_trials=params['hyperparameters']['trials'])

        logger.info("Best Params: %s", study.best_trial.params)
        logger.info("Best Score: %s", study.best_trial.value)
        logger.info(">>>>>>>> Hyperparameter-tuning finished")
        return study
    except Exception as e:
        raise CustomException("Error occured during fetching best hyperparams",
                              str(e))


def build_model(study, X_train, y_train):
    model = LGBMClassifier(**study.best_trial.params,
                           class_weight='balanced',
                           random_state=params['model_features']
                           ['random_state'],
                           n_jobs=-1)
    model.fit(X_train, y_train)
    return model


def save_model(model, save_path: str):
    try:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        full_path = os.path.join(save_path, "model.joblib")
        joblib.dump(model, full_path)
        logger.info(">>>>>>>> Model saved successfully at %s", full_path)
        return full_path

    except Exception as e:
        logger.error("Error occured during saving the model")
        raise CustomException("Failed to save model", str(e))


if __name__ == '__main__':
    try:
        mlflow.set_tracking_uri("file:mlruns")
        mlflow.set_experiment("hotel-reservation-exp")
        with mlflow.start_run() as run:
            with open("mlruns/current_run_id.txt", "w") as f:
                f.write(run.info.run_id)
            with Live() as live:
                logger.info(">>>>>>>> Model training started")
                logger.info(">>>>>>>> MLflow experimentation  started")

                logger.info(">>>>>>>> Tracking train dataset in mlflow")
                mlflow.log_artifact(sys.argv[1], artifact_path='datasets')

                logger.info(">>>>>>>> Reading train dataset")
                train_df = read_csv_files(sys.argv[1])

                X_train = train_df.drop(columns='booking_status')
                y_train = train_df["booking_status"]

                logger.info(">>>>>>>> train dataset read successfully")

                study = get_best_hyperparams(X_train, y_train)
                model = build_model(study, X_train, y_train)

                logger.info(
                    ">>>>>>>> Tracking best model params in mlflow and dvc")
                mlflow.log_params(model.get_params())

                for param_name, param_value in model.get_params().items():
                    live.log_param(param_name, param_value)

                model_path = save_model(model, save_path=sys.argv[2])

                logger.info(
                    ">>>>>>>> Tracking trained model in mlflow and dvc")
                mlflow.log_artifact(model_path, artifact_path='models')
                live.log_artifact(model_path, type='model')

                logger.info(">>>>>>>> Model training finished")

    except Exception as e:
        logger.error("Some error occurred during model training")
        raise CustomException("Some error occured", str(e))
