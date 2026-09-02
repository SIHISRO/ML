
from src.services.model_prediction import Model_prediction
from src.domain.config_entity import Model_prediction_config

class PredictionPipeline:
    def __init__(self, model_prediction_config: Model_prediction_config):
        self.model_prediction_config = model_prediction_config
        self.model_prediction_service = Model_prediction(model_prediction_config=model_prediction_config)

    def predict(self, input_data):
        return self.model_prediction_service.predict(input_data)
