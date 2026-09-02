from pathlib import Path

from src.pipelines.sift_prediction_pipeline import SiftPredictionPipeline

root = Path(__file__).resolve().parent
pipeline = SiftPredictionPipeline(output_dir=str(root / "temp" / "sift"))
result = pipeline.predict({
    "reference": str(root / "data" / "a.png"),
    "source": str(root / "data" / "b.png"),
})

print(result)