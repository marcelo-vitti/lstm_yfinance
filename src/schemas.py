from pydantic import BaseModel, Field
from typing import List


class PredictionRequest(BaseModel):
    close_prices: List[float] = Field(
        ...,
        description="Lista de preços de fechamento (Close). Deve conter LOOKBACK + 1 valores.",
    )


class PredictionResponse(BaseModel):
    predicted_log_return: float
    predicted_price: float
