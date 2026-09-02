import torch
import torch.nn as nn
from torchvision import models
import os
import json
import cv2
import numpy as np
import pandas as pd
import pandas.errors
from torch.utils.data import DataLoader,Dataset
from dataclasses import dataclass
import torch.nn.functional as F
from dataclasses import dataclass, field
from typing import Optional
from torchvision import transforms
from tqdm.auto import tqdm
import random
from src.domain.config_entity import Config


class LunarData(Dataset):

    def __init__(self, config):
        self.config = config
        self.data_files_path = config.data_files_path
        self.data_folder_name = config.data_folder_name
        self.image_size = config.image_size
        self.shift = config.homography_shift

        os.makedirs(self.data_folder_name, exist_ok=True)

        if os.path.exists(self.data_files_path):
            self.data_files = [
                x for x in os.listdir(self.data_files_path)
                if x.lower().endswith(".png")
            ]
            self.data_files = random.sample(self.data_files, 5000)
        else:
            self.data_files = []

        self.csv_path = os.path.join(
            self.data_folder_name,
            "dataset.csv"
        )

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(self.config.image_channels, 1, 1)),
            transforms.Normalize(
                mean=[0.5] * self.config.image_channels,
                std=[0.5] * self.config.image_channels
            )
        ])

        self.photometric_transform = transforms.ColorJitter(
            brightness=self.config.brightness,
            contrast=self.config.contrast
        )

        if os.path.exists(self.csv_path):
            try:
                self.df = pd.read_csv(self.csv_path)
            except pd.errors.EmptyDataError:
                self.df = self.build_dataset()
        else:
            self.df = self.build_dataset()

    def resize_with_padding(self, image):
        return cv2.resize(image, (self.image_size, self.image_size))

    def load_image(self, path):
        image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read: {path}")
        return self.resize_with_padding(image)

    def get_random_homography(self):
        size = self.image_size
        src = np.float32([
            [0, 0],
            [size - 1, 0],
            [size - 1, size - 1],
            [0, size - 1]
        ])

        dst = src + np.random.uniform(
            -self.shift, self.shift, (4, 2)
        ).astype(np.float32)

        return cv2.getPerspectiveTransform(src, dst).astype(np.float32)

    def build_dataset(self):
        rows = []
        for filename in self.data_files:
            path = os.path.join(self.data_files_path, filename)
            reference = self.load_image(path)
            h_ref_to_src = self.get_random_homography()

            source = cv2.warpPerspective(
                reference,
                h_ref_to_src,
                (self.image_size, self.image_size)
            )

            h_src_to_ref = np.linalg.inv(h_ref_to_src).astype(np.float32)
            name = os.path.splitext(filename)[0]

            reference_name = f"{name}_reference.png"
            source_name = f"{name}_source.png"

            cv2.imwrite(os.path.join(self.data_folder_name, reference_name), reference)
            cv2.imwrite(os.path.join(self.data_folder_name, source_name), source)

            rows.append({
                "reference_path": reference_name,
                "source_path": source_name,
                "h": json.dumps(h_src_to_ref.tolist())
            })

        df = pd.DataFrame(rows)
        df.to_csv(self.csv_path, index=False)
        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        reference = cv2.imread(
            os.path.join(self.data_folder_name, row["reference_path"]),
            cv2.IMREAD_GRAYSCALE
        )
        source = cv2.imread(
            os.path.join(self.data_folder_name, row["source_path"]),
            cv2.IMREAD_GRAYSCALE
        )

        reference = np.expand_dims(reference, axis=2)
        source = np.expand_dims(source, axis=2)

        reference_tensor = self.transform(reference)
        source_tensor = self.transform(source)

        source_tensor = self.photometric_transform(source_tensor)
        h = torch.tensor(json.loads(row["h"]), dtype=torch.float32)
        return reference_tensor, source_tensor, h

    def get_item(self, ref_path, src_path):
        ref_img = cv2.imread(ref_path, cv2.IMREAD_GRAYSCALE)
        src_img = cv2.imread(src_path, cv2.IMREAD_GRAYSCALE)

        if ref_img is None:
            raise ValueError(f"Could not read reference image: {ref_path}")
        if src_img is None:
            raise ValueError(f"Could not read source image: {src_path}")

        processed_ref = self.resize_with_padding(ref_img)
        processed_src = self.resize_with_padding(src_img)

        processed_ref = np.expand_dims(processed_ref, axis=2)
        processed_src = np.expand_dims(processed_src, axis=2)

        reference_tensor = self.transform(processed_ref)
        source_tensor = self.transform(processed_src)

        return reference_tensor, source_tensor


class FeatureEncoder(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        self.backbone = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3
        )

        self.projection = nn.Conv2d(
            256,
            config.encoder_out_channels,
            kernel_size=1
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.projection(x)
        return x


def dlt_solve(src_pts, dst_pts):
    B = src_pts.shape[0]
    A = torch.zeros((B, 8, 8), dtype=torch.float32, device=src_pts.device)
    b = torch.zeros((B, 8, 1), dtype=torch.float32, device=src_pts.device)

    for i in range(4):
        x_s, y_s = src_pts[:, i, 0], src_pts[:, i, 1]
        x_d, y_d = dst_pts[:, i, 0], dst_pts[:, i, 1]

        A[:, 2 * i, 0] = x_s
        A[:, 2 * i, 1] = y_s
        A[:, 2 * i, 2] = 1.0
        A[:, 2 * i, 6] = -x_s * x_d
        A[:, 2 * i, 7] = -y_s * x_d
        b[:, 2 * i, 0] = x_d

        A[:, 2 * i + 1, 3] = x_s
        A[:, 2 * i + 1, 4] = y_s
        A[:, 2 * i + 1, 5] = 1.0
        A[:, 2 * i + 1, 6] = -x_s * y_d
        A[:, 2 * i + 1, 7] = -y_s * y_d
        b[:, 2 * i + 1, 0] = y_d

    h_8 = torch.linalg.solve(A, b).squeeze(-1)
    ones = torch.ones((B, 1), dtype=torch.float32, device=src_pts.device)
    h_9 = torch.cat([h_8, ones], dim=-1)
    return h_9.view(B, 3, 3)


class RegistrationNetwork(nn.Module):

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.encoder = FeatureEncoder(config)

        self.regressor = nn.Sequential(
            nn.Conv2d(config.encoder_out_channels * 2, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, 8)
        )

    def forward(self, reference, source):
        feat_ref = self.encoder(reference)
        feat_src = self.encoder(source)

        concat_feat = torch.cat([feat_ref, feat_src], dim=1)
        delta = self.regressor(concat_feat).view(-1, 4, 2)

        size = float(self.config.image_size)
        src_corners = torch.tensor([
            [0.0, 0.0],
            [size - 1.0, 0.0],
            [size - 1.0, size - 1.0],
            [0.0, size - 1.0]
        ], dtype=torch.float32, device=reference.device).unsqueeze(0).repeat(reference.shape[0], 1, 1)

        dst_corners = src_corners + delta
        H_matrix = dlt_solve(src_corners, dst_corners)

        return {
            "feature_ref": feat_ref,
            "feature_src": feat_src,
            "delta": delta,
            "H": H_matrix
        }


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

