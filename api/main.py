from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes.predict_route import router as predict_router

app = FastAPI(title="Image Matching API", version="1.0.0")


origins = [
   "*"  # Allow all origins for development; restrict in production
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,  # List of allowed origins
    # allow_credentials=True,  # Allow cookies/authorization headers
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

app.include_router(predict_router, prefix="/api/v1", tags=["Prediction"])