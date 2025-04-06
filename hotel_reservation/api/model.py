import joblib

from hotel_reservation.config import config

MODEL = {'prediction': None}


def load_model():
    if config.load_weights:
        MODEL['prediction'] = joblib.load(
            'hotel_reservation/models/model.joblib')
