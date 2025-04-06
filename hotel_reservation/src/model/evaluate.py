import sys
from typing import Dict

import joblib
import mlflow
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from dvclive import Live
from hotel_reservation.src.logger import get_logger
from hotel_reservation.utils import read_csv_files
from hotel_reservation.utils.custom_exception import CustomException

logger = get_logger(__name__)


def evaluate(model, X_test: pd.DataFrame, y_test: pd.DataFrame) -> Dict:
    logger.info(">>>>>>>> Model evaluation started")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc_score = roc_auc_score(y_test, y_pred)

    logger.info(">>>>>>>> accuracy is: %s", accuracy)
    logger.info(">>>>>>>> precision is: %s", precision)
    logger.info(">>>>>>>> recall is: %s", recall)
    logger.info(">>>>>>>> f1-score is: %s", f1)
    logger.info(">>>>>>>> auc-score is: %s", auc_score)

    logger.info(">>>>>>>> Model evaluation finished")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1-score": f1,
        "auc_score": auc_score
    }


if __name__ == '__main__':
    try:
        mlflow.set_tracking_uri("file:mlruns")
        mlflow.set_experiment("hotel-reservation-exp")
        with open("mlruns/current_run_id.txt", "r") as f:
            run_id = f.read().strip()
        with mlflow.start_run(run_id=run_id):
            logger.info(">>>>>>>> Tracking test dataset in mlflow")
            mlflow.log_artifact(sys.argv[2], artifact_path='datasets')

            logger.info(">>>>>>>> Reading test dataset")
            test_df = read_csv_files(sys.argv[2])

            X_test = test_df.drop(columns='booking_status')
            y_test = test_df["booking_status"]

            model = joblib.load(sys.argv[1])
            metrics = evaluate(model=model, X_test=X_test, y_test=y_test)

            logger.info(
                ">>>>>>>> Tracking evalution metrics on test set in mlflow")
            mlflow.log_metrics(metrics)
            with Live() as live:
                for metric_name, metric_value in metrics.items():
                    live.log_metric(metric_name, metric_value)

    except Exception as e:
        logger.error("Some error occurred during model training")
        raise CustomException("Some error occured", str(e))
