from src.services.model_prediction import Model_prediction
from src.domain.config_entity import Model_prediction_config
from huggingface_hub import hf_hub_download
from kornia.feature import LoFTR
import cv2
import torch
import numpy as np

class PredictionPipeline:
    def __init__(self, model_prediction_config: Model_prediction_config):
        self.model_prediction_config = model_prediction_config
        self.model_prediction_service = Model_prediction(model_prediction_config=model_prediction_config)
        self.model = LoFTR(pretrained="outdoor")
        repo_id = "VashuTheGreat2/lunar-loftr-registration"

        weights_path = hf_hub_download(repo_id=repo_id, filename="best_lunar_loftr.pt")
        checkpoint = torch.load(weights_path, map_location="cpu")
        state_dict = checkpoint["model_state_dict"] if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint else checkpoint
        cleaned_state_dict = {
            k.replace("loftr.", "", 1) if k.startswith("loftr.") else k: v
            for k, v in state_dict.items()
        }
        self.model.load_state_dict(cleaned_state_dict, strict=False)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)

        if hasattr(self.model, "coarse_matching"):
            self.model.coarse_matching.thr = 0.01

    def preprocess_lunar_image(self, img_path, target_size=(480, 480), device="cuda"):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Image not found at path: {img_path}")

        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img = clahe.apply(img)

        img = cv2.resize(img, (target_size[1], target_size[0]))
        t_img = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) / 255.0
        return t_img.to(device)

    def compute_robust_homography_and_visuals(
        self,
        img0: np.ndarray,
        img1: np.ndarray,
        pred: dict,
        conf_thresh: float = 0.01,
        max_draw_matches: int = 80
    ):
        kpts0 = pred["keypoints0"].cpu().numpy()
        kpts1 = pred["keypoints1"].cpu().numpy()
        conf = pred["confidence"].cpu().numpy()

        vis_img0 = img0.copy()
        vis_img1 = img1.copy()

        if len(kpts0) < 4:
            print(f"[Warning] Total raw matches found by model: {len(kpts0)}. Minimum 4 required.")
            h0, w0 = img0.shape[:2]
            h1, w1 = img1.shape[:2]
            vis_matches = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
            vis_matches[:h0, :w0] = img0
            vis_matches[:h1, w0:w0 + w1] = img1
            return None, np.zeros((len(kpts0),), dtype=bool), vis_img0, vis_img1, vis_matches

        conf_mask = conf >= conf_thresh
        filtered_indices = np.where(conf_mask)[0]

        if len(filtered_indices) < 4:
            filtered_indices = np.argsort(conf)[-min(50, len(conf)):]

        pts0 = kpts0[filtered_indices]
        pts1 = kpts1[filtered_indices]

        H, inlier_mask = cv2.findHomography(
            pts0,
            pts1,
            method=cv2.USAC_MAGSAC,
            ransacReprojThreshold=5.0,
            maxIters=5000,
            confidence=0.99
        )

        if inlier_mask is None:
            inlier_mask = np.zeros((len(pts0),), dtype=bool)
        else:
            inlier_mask = inlier_mask.ravel().astype(bool)

        for pt in kpts0:
            cv2.circle(vis_img0, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)
        for pt in kpts1:
            cv2.circle(vis_img1, (int(pt[0]), int(pt[1])), 2, (0, 0, 255), -1)

        for pt in pts0[inlier_mask]:
            cv2.circle(vis_img0, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)
        for pt in pts1[inlier_mask]:
            cv2.circle(vis_img1, (int(pt[0]), int(pt[1])), 3, (0, 255, 0), -1)

        h0, w0 = img0.shape[:2]
        h1, w1 = img1.shape[:2]
        vis_matches = np.zeros((max(h0, h1), w0 + w1, 3), dtype=np.uint8)
        vis_matches[:h0, :w0] = img0
        vis_matches[:h1, w0:w0 + w1] = img1

        step = max(1, len(filtered_indices) // max_draw_matches)
        for i in range(0, len(filtered_indices), step):
            pt_a = (int(pts0[i][0]), int(pts0[i][1]))
            pt_b = (int(pts1[i][0]) + w0, int(pts1[i][1]))
            color = (0, 255, 0) if inlier_mask[i] else (0, 0, 255)

            cv2.line(vis_matches, pt_a, pt_b, color, 1, cv2.LINE_AA)
            cv2.circle(vis_matches, pt_a, 2, color, -1)
            cv2.circle(vis_matches, pt_b, 2, color, -1)

        return H, inlier_mask, vis_img0, vis_img1, vis_matches

    def register_and_warp(self, img0_np, img1_np, H):
        if H is None:
            return img1_np.copy(), img0_np.copy()

        H_inv = np.linalg.inv(H)
        h, w = img0_np.shape[:2]

        warped_img1 = cv2.warpPerspective(img1_np, H_inv, (w, h))
        overlay = cv2.addWeighted(img0_np, 0.5, warped_img1, 0.5, 0)
        return warped_img1, overlay

    def predict(self, input_data):
        self.model.eval()

        with torch.no_grad():
            matches = self.model({
                "image0": self.preprocess_lunar_image(img_path=input_data['reference'], device=self.device),
                "image1": self.preprocess_lunar_image(img_path=input_data['source'], device=self.device)
            })

        print(f"Raw Matches Returned by LoFTR: {len(matches['keypoints0'])}")

        img0 = cv2.imread(input_data['reference'], cv2.IMREAD_COLOR)
        img1 = cv2.imread(input_data['source'], cv2.IMREAD_COLOR)

        img0 = cv2.resize(img0, (480, 480))
        img1 = cv2.resize(img1, (480, 480))

        H, inlier_mask, vis_img0, vis_img1, vis_matches = self.compute_robust_homography_and_visuals(
            img0=img0,
            img1=img1,
            pred=matches,
            conf_thresh=0.01,
            max_draw_matches=80
        )

        wraped_img1, overlay = self.register_and_warp(img0_np=img0, img1_np=img1, H=H)
        return matches, H, inlier_mask, vis_img0, vis_img1, vis_matches, wraped_img1, overlay