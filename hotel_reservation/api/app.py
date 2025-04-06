from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse

from hotel_reservation.api.model import load_model
from hotel_reservation.api.templates import templates

from .routers import prediction


@asynccontextmanager
# pylint: disable=unused-argument
async def lifespan(appl: FastAPI):
    load_model()
    yield


app = FastAPI(title="Hotel Reservation Cancellation Prediction",
              lifespan=lifespan)


@app.get("/healthcheck", tags=['Monitoring'])
async def healthcheck():
    return ""


@app.get("/", response_class=HTMLResponse, tags=["Homepage"])
async def homepage(request: Request):
    return templates.TemplateResponse(request,
                                      name="homepage.html",
                                      context={})


app.include_router(prediction.router)
