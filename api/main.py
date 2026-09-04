from fastapi import FastAPI

from api.routes.predict_route import router as predict_router

app = FastAPI(title="Image Matching API", version="1.0.0")


app.include_router(predict_router, prefix="/api/v1", tags=["Prediction"])