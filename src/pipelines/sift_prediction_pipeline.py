from src.services.predict_sift import PredictSift


class SiftPredictionPipeline:
    def __init__(self, output_dir="temp/sift"):
        self.predict_sift = PredictSift(output_dir=output_dir)

    def predict(self, input_data):
        return self.predict_sift.predict(input_data)