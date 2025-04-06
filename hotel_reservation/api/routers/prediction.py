from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
import numpy as np

from hotel_reservation.api.model import MODEL
from hotel_reservation.api.templates import templates

router = APIRouter(prefix="/predict", tags=["Prediction"])


@router.post(
    '',
    response_class=HTMLResponse,
    summary='Predict the cancellation of a hotel based on user inputs')
async def get_prediction(request: Request):
    form = await request.form()

    lead_time = int(form["lead_time"])
    avg_price_per_room = float(form["avg_price_per_room"])
    no_of_special_requests = int(form["no_of_special_requests"])
    no_of_week_nights = int(form["no_of_week_nights"])
    no_of_weekend_nights = int(form["no_of_weekend_nights"])
    no_of_adults = int(form["no_of_adults"])
    market_segment_type_Offline = int(form["market_segment_type_Offline"])
    market_segment_type_Online = 1 - market_segment_type_Offline

    features = np.array([[
        lead_time, avg_price_per_room, no_of_special_requests,
        no_of_week_nights, market_segment_type_Online, no_of_weekend_nights,
        market_segment_type_Offline, no_of_adults
    ]])

    prediction = MODEL['prediction'].predict(features)

    return templates.TemplateResponse(
        request=request,
        name="homepage.html",
        context={'prediction': int(prediction[0])})
