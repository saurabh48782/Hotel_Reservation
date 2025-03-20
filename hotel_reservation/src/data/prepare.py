import os
import sys
from typing import List

import dvc.api
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from hotel_reservation import utils
from hotel_reservation.src.logger import get_logger
from hotel_reservation.utils.custom_exception import CustomException

logger = get_logger(__name__)


class DataPreparation:

    def __init__(self, raw_filepath: str, processed_dir: str):
        self.raw_filepath = raw_filepath
        self.processed_dir = processed_dir

        if not os.path.exists(processed_dir):
            os.makedirs(processed_dir)

    def prepare(self):
        params = dvc.api.params_show()

        try:
            logger.info(">>>>>>>> Reading raw csv file")
            df = utils.read_csv_files(self.raw_filepath)
        except Exception as e:
            raise CustomException("Error in reading raw CSV file", e) from e

        try:
            logger.info(">>>>>>>> Data cleaning started")
            df = self.data_cleaning(df)
            logger.info(">>>>>>>> Data cleaning finished")
        except Exception as e:
            raise CustomException("Error in data cleaning step", e) from e

        try:
            logger.info(">>>>>>>> Feature engineering started")
            df = self.feature_eng(df, params)
            logger.info(">>>>>>>> Feature engineering finished")
        except Exception as e:
            raise CustomException("Error in feature engineering step",
                                  e) from e

        try:
            logger.info(">>>>>>>> Data splitting started")

            df.sort_values('arrival_full_date',
                           ascending=True,
                           inplace=True,
                           ignore_index=True)

            train_data = df.loc[(
                df['arrival_full_date']
                < params['split_criterion']['train_split'])].reset_index(
                    drop=True)
            test_data = df.loc[df['arrival_full_date'] >=
                               params['split_criterion']
                               ['train_split']].reset_index(drop=True)
            train_data.drop(columns=[
                'arrival_year', 'arrival_month', 'arrival_date',
                'arrival_full_date'
            ],
                            axis=1,
                            inplace=True)
            test_data.drop(columns=[
                'arrival_year', 'arrival_month', 'arrival_date',
                'arrival_full_date'
            ],
                           axis=1,
                           inplace=True)

            logger.info(">>>>>>>> Data splitting finished")
        except Exception as e:
            raise CustomException("Error in data splitting step", e) from e

        try:
            logger.info(">>>>>>>> Feature selection started")

            top_features = self.get_best_features(train_data, params)

            logger.info(">>>>>>>> Feature selection completed")
        except Exception as e:
            raise CustomException("Error in feature selection step", e) from e

        train_data = train_data[top_features + ["booking_status"]]
        test_data = test_data[top_features + ["booking_status"]]

        train_data.to_csv(os.path.join(self.processed_dir, 'train.csv'),
                          index=False,
                          header=True)
        test_data.to_csv(os.path.join(self.processed_dir, 'test.csv'),
                         index=False,
                         header=True)

        logger.info(
            ">>>>>>>> Splitted data saved successfully in preprocessed folder")

    def data_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        '''Drop/Delete irrelevant data'''

        df.drop('Booking_ID', axis=1, inplace=True)
        df.drop_duplicates(keep='first', ignore_index=True, inplace=True)
        return df.loc[~((df['arrival_date'] == 29) &
                        (df['arrival_month'] == 2))].reset_index(drop=True)

    def feature_eng(self, df: pd.DataFrame, params) -> pd.DataFrame:
        df['arrival_full_date'] = pd.to_datetime(
            df[['arrival_year', 'arrival_month', 'arrival_date']].rename(
                columns={
                    'arrival_year': 'year',
                    'arrival_month': 'month',
                    'arrival_date': 'day'
                }))

        df['arrival_quarter'] = df['arrival_full_date'].dt.quarter.map({
            1:
            "Jan-Mar",
            2:
            "Apr-Jun",
            3:
            "Jul-Sep",
            4:
            "Oct-Dec"
        })

        df['is_weekend'] = df['arrival_full_date'].dt.weekday >= 5
        df['is_weekend'] = df['is_weekend'].astype(int)

        df = pd.get_dummies(
            df,
            columns=params['model_features']['encode_features'],
            drop_first=True,
            dtype=int)
        df['booking_status'] = df['booking_status'].map({
            'Not_Canceled': 0,
            'Canceled': 1
        })
        return df

    def get_best_features(self, df: pd.DataFrame, params) -> List[str]:
        """Returns a list of eight best model features"""

        independent_feats = df.drop(columns='booking_status')
        target = df["booking_status"]

        model = RandomForestClassifier(
            random_state=params['split_criterion']['seed'],
            class_weight='balanced')
        model.fit(independent_feats, target)

        feature_importance_df = pd.DataFrame({
            'feature':
            independent_feats.columns,
            'importance':
            model.feature_importances_
        })

        features_importance_df = feature_importance_df.sort_values(
            by="importance", ascending=False).reset_index(drop=True)

        logger.info("Best features are: %s",
                    features_importance_df['feature'].head(8).values.tolist())

        return features_importance_df["feature"].head(8).values.tolist()

    def run(self):
        try:
            logger.info(">>>>>>>> Data preparation started")
            self.prepare()
            logger.info(">>>>>>>> Data preparation completed successfully")
        except Exception as ce:
            logger.error(
                CustomException("Error occured while preprocessing data",
                                str(ce)))


if __name__ == '__main__':
    preprocess = DataPreparation(raw_filepath=sys.argv[1],
                                 processed_dir=sys.argv[2])
    preprocess.run()
