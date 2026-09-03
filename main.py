import os
import shutil
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from src.domain.config_entity import Model_prediction_config
from src.pipelines.prediction_pipeline import PredictionPipeline

app = FastAPI(title="ISRO Registration Network API")

# Add CORS so frontend can easily hit this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

root = Path(__file__).resolve().parent

# Mount the temp directory so images can be accessed via URL
temp_dir = root / "temp"
temp_dir.mkdir(parents=True, exist_ok=True)
app.mount("/temp", StaticFiles(directory=str(temp_dir)), name="temp")

# Initialize model pipeline once at startup
config = Model_prediction_config(
    model_path=str(root / "artifacts" / "trained_model" / "registration_model.pth"),
    mlflow_tracking_uri="https://dagshub.com/vanshsharma7832/sih_isro.mlflow",
    mlflow_model_uri="models:/RegistrationNetwork/latest",
    output_dir=str(temp_dir),
)
pipeline = PredictionPipeline(config)


@app.post("/predict")
async def predict_images(request: Request, reference: UploadFile = File(...), source: UploadFile = File(...)):
    try:
        # Create temp data directory if it doesn't exist
        data_dir = root / "data"
        data_dir.mkdir(parents=True, exist_ok=True)

        ref_path = data_dir / f"temp_{reference.filename}"
        src_path = data_dir / f"temp_{source.filename}"

        # Save uploaded files temporarily
        with open(ref_path, "wb") as buffer:
            shutil.copyfileobj(reference.file, buffer)
        with open(src_path, "wb") as buffer:
            shutil.copyfileobj(source.file, buffer)

        # Run Prediction
        input_data = {
            "reference": str(ref_path),
            "source": str(src_path),
        }
        result_paths = pipeline.predict(input_data)

        # Get the base URL of the server (e.g., http://localhost:8000)
        base_url = str(request.base_url)

        # Convert output image paths to local URLs instead of Base64
        image_urls = {}
        for key, path in result_paths.items():
            if os.path.exists(path):
                # Extract just the filename (e.g., 'overlay.png')
                filename = os.path.basename(path)
                # Create a URL like http://localhost:8000/temp/overlay.png
                image_urls[key] = f"{base_url}temp/{filename}"
            else:
                image_urls[key] = None

        # Clean up temporary uploaded files
        if os.path.exists(ref_path):
            os.remove(ref_path)
        if os.path.exists(src_path):
            os.remove(src_path)

        return {
            "status": "success",
            "data": image_urls
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
