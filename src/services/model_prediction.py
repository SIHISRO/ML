from src.domain.config_entity import Config, Model_prediction_config
from src.domain.model_acess import Model
from src.models.RegistrationNetwork import LunarData
import torch
import os
from pathlib import Path

class Model_prediction:
    def __init__(self,model_prediction_config: Model_prediction_config):
        self.config = model_prediction_config
        model_config = Config(
            device=self.config.device,
            save_path=self.config.model_path,
            mlflow_tracking_uri=self.config.mlflow_tracking_uri,
            mlflow_model_uri=self.config.mlflow_model_uri,
        )
        self.model = Model(config=model_config)
        self.data = LunarData(Config(data_files_path="__unused__"))
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.config.model_path and os.path.isfile(self.config.model_path):
            self.model.load(path=self.config.model_path)
        else:
            self.load_from_mlflow()

    def load_from_mlflow(self):
        if not self.config.mlflow_model_uri:
            raise FileNotFoundError(
                f"Model checkpoint was not found: {self.config.model_path}. "
                "Set mlflow_model_uri to load the model from MLflow."
            )

        import mlflow
        import mlflow.pytorch

        if self.config.mlflow_tracking_uri:
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)

        loaded_network = mlflow.pytorch.load_model(self.config.mlflow_model_uri, map_location=self.model.device)
        self.model.network = loaded_network.to(self.model.device)
        if self.config.model_path:
            os.makedirs(os.path.dirname(self.config.model_path), exist_ok=True)
            self.model.save(path=self.config.model_path)
        
    def predict(self, input_data):
        if not isinstance(input_data, dict):
            raise ValueError("Prediction input must be a dictionary.")
        if "reference" not in input_data or "source" not in input_data:
            raise ValueError("Prediction input must contain reference and source paths.")

        reference, source = self.data.get_item(
            str(input_data["reference"]),
            str(input_data["source"]),
        )
        reference = reference.unsqueeze(0)
        source = source.unsqueeze(0)
        with torch.no_grad():
            output = self.model.predict(reference, source)

        return self.save_visualizations(reference, source, output)

    def save_visualizations(self, reference, source, output):
        self.save_feature_maps(output["feature_ref"], output["feature_src"])
        self.save_feature_matches(output["feature_ref"], output["feature_src"])
        self.save_correlation(output["feature_ref"], output["feature_src"])
        self.save_homography(reference, source, output["H"])
        return {
            "feature_activation": str(self.output_dir / "feature_activations.png"),
            "feature_matches": str(self.output_dir / "feature_matches.png"),
            "feature_correlation": str(self.output_dir / "feature_correlation.png"),
            "homography_overlay": str(self.output_dir / "homography_overlay.png"),
            "reference": str(self.output_dir / "reference.png"),
            "source": str(self.output_dir / "source.png"),
            "registered_source": str(self.output_dir / "registered_source.png"),
            "overlay": str(self.output_dir / "overlay.png"),
        }

    def save_feature_maps(self, feature_ref, feature_src):
        import matplotlib.pyplot as plt
        import numpy as np

        map_ref = np.linalg.norm(feature_ref.detach().cpu().numpy()[0], axis=0)
        map_src = np.linalg.norm(feature_src.detach().cpu().numpy()[0], axis=0)
        lower = min(map_ref.min(), map_src.min())
        upper = max(map_ref.max(), map_src.max())
        figure, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].imshow(map_ref, cmap="turbo", vmin=lower, vmax=upper)
        axes[0].set_title("Reference feature activation")
        axes[1].imshow(map_src, cmap="turbo", vmin=lower, vmax=upper)
        axes[1].set_title("Source feature activation")
        for axis in axes:
            axis.axis("off")
        plt.tight_layout()
        plt.savefig(self.output_dir / "feature_activations.png", dpi=200, bbox_inches="tight")
        plt.close(figure)

    def save_feature_matches(self, feature_ref, feature_src):
        import matplotlib.pyplot as plt
        import numpy as np

        features_ref = feature_ref.detach().cpu().numpy()[0]
        features_src = feature_src.detach().cpu().numpy()[0]
        _, height, width = features_ref.shape
        descriptors_ref = features_ref.reshape(features_ref.shape[0], -1).T
        descriptors_src = features_src.reshape(features_src.shape[0], -1).T
        descriptors_ref /= np.linalg.norm(descriptors_ref, axis=1, keepdims=True) + 1e-8
        descriptors_src /= np.linalg.norm(descriptors_src, axis=1, keepdims=True) + 1e-8
        similarity = descriptors_ref @ descriptors_src.T
        forward = similarity.argmax(axis=1)
        backward = similarity.argmax(axis=0)
        matches = [(index, target) for index, target in enumerate(forward) if backward[target] == index]
        matches.sort(key=lambda pair: similarity[pair[0], pair[1]], reverse=True)
        matches = matches[:40]
        combined = np.concatenate([
            np.linalg.norm(features_ref, axis=0),
            np.linalg.norm(features_src, axis=0),
        ], axis=1)
        figure, axis = plt.subplots(figsize=(12, 6))
        axis.imshow(combined, cmap="turbo")
        for index, target in matches:
            y_ref, x_ref = divmod(index, width)
            y_src, x_src = divmod(target, width)
            axis.plot([x_ref, x_src + width], [y_ref, y_src], color="white", linewidth=0.6, alpha=0.65)
            axis.scatter([x_ref, x_src + width], [y_ref, y_src], color="red", s=8)
        axis.axvline(width - 0.5, color="white", linewidth=1)
        axis.set_title(f"Mutual feature matches on {height}x{width} grid: {len(matches)}")
        axis.axis("off")
        plt.tight_layout()
        plt.savefig(self.output_dir / "feature_matches.png", dpi=200, bbox_inches="tight")
        plt.close(figure)

    def save_correlation(self, feature_ref, feature_src):
        import matplotlib.pyplot as plt
        import numpy as np

        features_ref = feature_ref.detach().cpu().numpy()[0]
        features_src = feature_src.detach().cpu().numpy()[0]
        channels, height, width = features_ref.shape
        descriptors_ref = features_ref.reshape(channels, -1).T
        descriptors_src = features_src.reshape(channels, -1).T
        descriptors_ref /= np.linalg.norm(descriptors_ref, axis=1, keepdims=True) + 1e-8
        descriptors_src /= np.linalg.norm(descriptors_src, axis=1, keepdims=True) + 1e-8
        correlation = descriptors_ref @ descriptors_src.T
        spatial_correlation = np.diag(correlation).reshape(height, width)
        figure, axis = plt.subplots(figsize=(7, 7))
        image = axis.imshow(spatial_correlation, cmap="coolwarm", vmin=-1, vmax=1)
        axis.set_title(f"Feature correlation: {height}x{width}")
        axis.set_xlabel("Source feature column")
        axis.set_ylabel("Reference feature row")
        axis.set_xticks(range(width))
        axis.set_yticks(range(height))
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        plt.tight_layout()
        plt.savefig(self.output_dir / "feature_correlation.png", dpi=200, bbox_inches="tight")
        plt.close(figure)

    def save_homography(self, reference, source, homography):
        import cv2
        import matplotlib.pyplot as plt
        import numpy as np

        reference_image = reference.detach().cpu().numpy()[0, 0]
        source_image = source.detach().cpu().numpy()[0, 0]
        matrix = homography.detach().cpu().numpy()[0]
        reference_image = np.clip((reference_image + 1.0) * 127.5, 0, 255).astype(np.uint8)
        source_image = np.clip((source_image + 1.0) * 127.5, 0, 255).astype(np.uint8)
        warped_source = cv2.warpPerspective(source_image, matrix, (source_image.shape[1], source_image.shape[0]))
        overlay = 0.5 * reference_image + 0.5 * warped_source
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        cv2.imwrite(str(self.output_dir / "reference.png"), reference_image)
        cv2.imwrite(str(self.output_dir / "source.png"), source_image)
        cv2.imwrite(str(self.output_dir / "registered_source.png"), warped_source)
        cv2.imwrite(str(self.output_dir / "overlay.png"), overlay)
        figure, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes[0, 0].imshow(reference_image, cmap="gray")
        axes[0, 0].set_title("Reference")
        axes[0, 1].imshow(source_image, cmap="gray")
        axes[0, 1].set_title("Original source")
        axes[1, 0].imshow(warped_source, cmap="gray")
        axes[1, 0].set_title("Warped source")
        axes[1, 1].imshow(overlay, cmap="gray")
        axes[1, 1].set_title("Overlay")
        for axis in axes.flat:
            axis.axis("off")
        plt.tight_layout()
        plt.savefig(self.output_dir / "homography_overlay.png", dpi=200, bbox_inches="tight")
        plt.close(figure)