
from dataclasses import dataclass
from typing import Optional


@dataclass
class Config:
    data_files_path: str = "data"
    data_folder_name: str = "preprocessed_data"
    image_size: int = 256
    homography_shift: float = 40.0
    image_channels: int = 3
    brightness: float = 0.5
    contrast: float = 0.5
    encoder_out_channels: int = 128
    batch_size: int = 16
    lr: float = 1e-4
    epochs: int = 100
    device: Optional[str] = None
    save_path: str = "registration_model.pth"
    save_window: int = 5
    metric_path: str = "metrics.pkl"
    lr_scheduler_factor: float = 0.5
    lr_scheduler_patience: int = 5
    early_stopping_patience: int = 10
    early_stopping_min_delta: float = 0.001
    mlflow_tracking_uri: Optional[str] = None
    mlflow_model_uri: Optional[str] = None


@dataclass
class Model_prediction_config:
    model_path: str = "artifacts/trained_model/registration_model.pth"
    device: Optional[str] = None
    mlflow_tracking_uri: Optional[str] = None
    mlflow_model_uri: Optional[str] = None
    output_dir: str = "temp"