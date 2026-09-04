import base64
import cv2
import numpy as np
import torch
from pathlib import Path
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import JSONResponse

from api.middlewares.multer_middleware import multer_middleware
from src.pipelines.prediction_pipeline import PredictionPipeline
from src.domain.config_entity import Model_prediction_config

router = APIRouter()

prediction_pipeline = PredictionPipeline(model_prediction_config=Model_prediction_config())


def numpy_to_base64(img_np: np.ndarray, format: str = ".jpg") -> str:
    """NumPy image array ko base64 data URI string me convert karta hai."""
    if img_np is None:
        return ""
    success, buffer = cv2.imencode(format, img_np)
    if not success:
        return ""
    b64_str = base64.b64encode(buffer).decode("utf-8")
    mime = "image/jpeg" if format.lower() in [".jpg", ".jpeg"] else "image/png"
    return f"data:{mime};base64,{b64_str}"


@router.post("/predict")
async def predict(image_paths: list[str] = Depends(multer_middleware)):
    if len(image_paths) != 2:
        raise HTTPException(
            status_code=400,
            detail="Upload exactly two images as multipart form-data files.",
        )

    ref_img, src_img = image_paths
    matches, H, inlier_mask, vis_img0, vis_img1, vis_matches, warped_img1, overlay = (
        prediction_pipeline.predict({
            "reference": ref_img,
            "source": src_img,
        })
    )

    keypoints0 = matches["keypoints0"].cpu().numpy().tolist() if isinstance(matches.get("keypoints0"), torch.Tensor) else []
    keypoints1 = matches["keypoints1"].cpu().numpy().tolist() if isinstance(matches.get("keypoints1"), torch.Tensor) else []
    confidence = matches["confidence"].cpu().numpy().tolist() if isinstance(matches.get("confidence"), torch.Tensor) else []
    
    homography_matrix = H.tolist() if isinstance(H, np.ndarray) else None
    inliers = inlier_mask.tolist() if isinstance(inlier_mask, np.ndarray) else []

    response_payload = {
        "status": "success",
        "metrics": {
            "total_matches": len(keypoints0),
            "inliers_count": int(sum(inliers)) if inliers else 0,
        },
        "homography": homography_matrix,
        "keypoints": {
            "reference": keypoints0,
            "source": keypoints1,
            "confidence": confidence,
            "inlier_mask": inliers
        },
        "visualizations": {
            "ref_points": numpy_to_base64(vis_img0),
            "src_points": numpy_to_base64(vis_img1),
            "match_lines": numpy_to_base64(vis_matches),
            "warped_source": numpy_to_base64(warped_img1),
            "registered_overlay": numpy_to_base64(overlay),
        }
    }


    output_dir = Path(__file__).resolve().parents[2]
    images_to_save = {
        "vis_img0.jpg": vis_img0,
        "vis_img1.jpg": vis_img1,
        "vis_matches.jpg": vis_matches,
        "warped_img1.jpg": warped_img1,
        "overlay.jpg": overlay,
    }
    for filename, image in images_to_save.items():
        saved = cv2.imwrite(str(output_dir / filename), image)
        if not saved:
            raise HTTPException(status_code=500, detail=f"Could not save {filename}.")

    return JSONResponse(content=response_payload, status_code=200)