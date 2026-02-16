from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
from typing import List

app = FastAPI()

# -------- REQUEST MODEL --------
class DataPoint(BaseModel):
    ds: str   # date string
    y: float  # usage value

class ForecastRequest(BaseModel):
    history: List[DataPoint]   # historical real data
    forecast_periods: int = 6  # months ahead


# -------- FORECAST FUNCTION --------
def run_prophet_forecast(df: pd.DataFrame, periods=6):

    df['ds'] = pd.to_datetime(df['ds'])

    model = Prophet(
        growth="linear",
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        changepoint_prior_scale=0.3
    )

    model.fit(df)

    future = model.make_future_dataframe(periods=periods, freq='MS')  # Monthly Start
    forecast = model.predict(future)
    forecast['yhat'] = forecast['yhat'].clip(lower=0)

    # Metrics (training overlap)
    metric_df = forecast.set_index('ds')[['yhat']].join(df.set_index('ds')[['y']], how='inner')
    mae = mean_absolute_error(metric_df['y'], metric_df['yhat'])
    rmse = np.sqrt(mean_squared_error(metric_df['y'], metric_df['yhat']))
    mape = np.mean(np.abs((metric_df['y'] - metric_df['yhat']) / (metric_df['y'] + 1))) * 100

    result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)

    return {
        "metrics": {
            "mae": round(mae, 2),
            "rmse": round(rmse, 2),
            "mape": round(mape, 2)
        },
        "forecast": result.to_dict(orient='records')
    }


# -------- API ENDPOINT --------
@app.post("/forecast")
async def generate_forecast(request: ForecastRequest):
    try:
        df = pd.DataFrame([{"ds": d.ds, "y": d.y} for d in request.history])

        if len(df) < 6:
            raise HTTPException(status_code=400, detail="Not enough historical data")

        results = run_prophet_forecast(df, periods=request.forecast_periods)

        return {
            "status": "success",
            "data": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
