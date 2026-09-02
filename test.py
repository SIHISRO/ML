from pathlib import Path

from src.domain.config_entity import Model_prediction_config
from src.pipelines.prediction_pipeline import PredictionPipeline

root = Path(__file__).resolve().parent
config = Model_prediction_config(
    model_path=str(root / "artifacts" / "trained_model" / "registration_model.pth"),
    mlflow_tracking_uri="https://dagshub.com/vanshsharma7832/sih_isro.mlflow",
    mlflow_model_uri="models:/RegistrationNetwork/latest",
    output_dir=str(root / "temp"),
)
pipeline = PredictionPipeline(config)
result = pipeline.predict({
    "reference": str(root / "data" / "a.png"),
    "source": str(root / "data" / "b.png"),
})

print(result)
