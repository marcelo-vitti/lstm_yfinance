import time

from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator

from src.schemas import PredictionRequest, PredictionResponse
from src.inference import predict_from_close
from src.utils.logging import log_resource_usage

app = FastAPI(
    title="VISA Stock Predictor",
    description="LSTM model trained on log-returns. API accepts Close prices.",
    version="1.0.0",
)


Instrumentator().instrument(app).expose(app)


@app.get("/")
def health():
    return {"status": "ok", "model": "lstm_visa_vus"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):

    try:
        start = time.time()
        log_ret, price = predict_from_close(request.close_prices)
        latency = time.time() - start
        log_resource_usage()

        return PredictionResponse(
            predicted_log_return=log_ret,
            predicted_price=price,
            latency_ms=latency * 1000,
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail="Prediction failed")
