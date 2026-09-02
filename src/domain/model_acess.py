from src.domain.config_entity import Config
from typing import Optional
from src.models.RegistrationNetwork import RegistrationNetwork
import torch
from tqdm.auto import tqdm
import os
from torch.nn import functional as F

def corner_loss(H_pred, H_true, size=256):
    corners = torch.tensor([
        [0.0, 0.0, 1.0],
        [size - 1.0, 0.0, 1.0],
        [size - 1.0, size - 1.0, 1.0],
        [0.0, size - 1.0, 1.0]
    ], dtype=torch.float32, device=H_pred.device).unsqueeze(0).repeat(H_pred.shape[0], 1, 1).transpose(1, 2)

    pts_pred = torch.bmm(H_pred, corners)
    pts_pred = pts_pred[:, :2, :] / (pts_pred[:, 2:, :] + 1e-8)

    pts_true = torch.bmm(H_true, corners)
    pts_true = pts_true[:, :2, :] / (pts_true[:, 2:, :] + 1e-8)

    return F.l1_loss(pts_pred / size, pts_true / size)


class Model:

    def __init__(self, config: Optional[Config] = None):
        self.config = config or Config()

        self.device = self.config.device or (
            "cuda" if torch.cuda.is_available() else "cpu"
        )

        self.network = RegistrationNetwork(self.config).to(self.device)
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.config.lr
        )

        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=self.config.lr_scheduler_factor,
            patience=self.config.lr_scheduler_patience
        )

        self.train_loss = []
        self.val_loss = []

    def loss(self, pred, target):
        return corner_loss(pred, target, size=self.config.image_size)

    def train(self, train_loader, valid_loader=None, epochs=None, train_new=False, specific_path=None):
        if not train_new:
            if os.path.exists(self.config.save_path) and not specific_path:
                print("loading saved model")
                self.load()
            elif specific_path is not None and os.path.exists(specific_path):
                print("loading saved model from specific_path")
                self.load(path=specific_path)

        num_epochs = epochs or self.config.epochs
        best_loss_for_saving = float("inf")
        epochs_no_improve_for_saving = 0

        for epoch in tqdm(range(num_epochs)):
            self.network.train()
            total_loss = 0.0

            for reference, source, H_true in tqdm(train_loader, leave=False):
                reference = reference.to(self.device)
                source = source.to(self.device)
                H_true = H_true.to(self.device)

                output = self.network(reference, source)
                H_pred = output["H"]

                loss = self.loss(H_pred, H_true)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            train_loss = total_loss / len(train_loader)
            self.train_loss.append(train_loss)
            current_monitored_loss = train_loss

            if valid_loader is not None:
                valid_loss = self.validate(valid_loader)
                self.val_loss.append(valid_loss)
                print(f"Epoch {epoch + 1}/{num_epochs} Train: {train_loss:.6f} Valid: {valid_loss:.6f}")
                self.scheduler.step(valid_loss)
                current_monitored_loss = valid_loss
            else:
                print(f"Epoch {epoch + 1}/{num_epochs} Train: {train_loss:.6f}")

            if current_monitored_loss < best_loss_for_saving - self.config.early_stopping_min_delta:
                best_loss_for_saving = current_monitored_loss
                epochs_no_improve_for_saving = 0
                self.save()
                print(f"Model Saved: Monitored loss improved to {best_loss_for_saving:.6f}")
            else:
                epochs_no_improve_for_saving += 1
                print(f"Monitored loss did not improve. Epochs without improvement: {epochs_no_improve_for_saving}/{self.config.early_stopping_patience}")

            if epochs_no_improve_for_saving >= self.config.early_stopping_patience:
                print(f"Early stopping triggered after {self.config.early_stopping_patience} epochs without improvement.")
                break

    def validate(self, valid_loader):
        self.network.eval()
        total_loss = 0.0

        with torch.no_grad():
            for reference, source, H_true in valid_loader:
                reference = reference.to(self.device)
                source = source.to(self.device)
                H_true = H_true.to(self.device)

                output = self.network(reference, source)
                loss = self.loss(output["H"], H_true)
                total_loss += loss.item()

        return total_loss / len(valid_loader)

    @torch.no_grad()
    def predict(self, reference, source):
        self.network.eval()
        reference = reference.to(self.device)
        source = source.to(self.device)
        return self.network(reference, source)

    def save(self, path=None):
        save_path = path or self.config.save_path
        torch.save(self.network.state_dict(), save_path)
        with open(self.config.metric_path, "wb") as f:
            import pickle
            pickle.dump({"train_loss": self.train_loss, "val_loss": self.val_loss}, f)

    def load(self, path=None):
        load_path = path or self.config.save_path
        self.network.load_state_dict(
            torch.load(load_path, map_location=self.device)
        )
        if os.path.exists(self.config.metric_path):
            import pickle
            obj = pickle.load(open(self.config.metric_path, "rb"))
            self.train_loss = obj.get('train_loss', [])
            self.val_loss = obj.get('val_loss', [])