from pathlib import Path

import cv2
import numpy as np


class PredictSift:
    def __init__(self, output_dir="temp", ratio_threshold=0.75, ransac_threshold=5.0):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.ratio_threshold = ratio_threshold
        self.ransac_threshold = ransac_threshold

    def predict(self, input_data):
        if not isinstance(input_data, dict):
            raise ValueError("Prediction input must be a dictionary.")
        if "reference" not in input_data or "source" not in input_data:
            raise ValueError("Prediction input must contain reference and source paths.")

        reference = self.load_image(input_data["reference"])
        source = self.load_image(input_data["source"])
        keypoints_ref, descriptors_ref = self.detect(reference)
        keypoints_src, descriptors_src = self.detect(source)
        good_matches = self.match(descriptors_ref, descriptors_src)
        homography, inlier_mask = self.find_homography(keypoints_ref, keypoints_src, good_matches)

        self.save_images(
            reference,
            source,
            keypoints_ref,
            keypoints_src,
            good_matches,
            inlier_mask,
            homography,
        )

        return {
            "feature_activation": str(self.output_dir / "feature_activations.png"),
            "feature_matches": str(self.output_dir / "feature_matches.png"),
            "feature_correlation": str(self.output_dir / "feature_correlation.png"),
            "homography_overlay": str(self.output_dir / "homography_overlay.png"),
            "reference": str(self.output_dir / "reference.png"),
            "source": str(self.output_dir / "source.png"),
            "registered_source": str(self.output_dir / "registered_source.png"),
            "overlay": str(self.output_dir / "overlay.png"),
            "matches": len(good_matches),
            "inliers": int(inlier_mask.sum()) if inlier_mask is not None else 0,
            "homography": homography.tolist() if homography is not None else None,
        }

    def load_image(self, path):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Image was not found: {path}")
        return image

    def detect(self, image):
        sift = cv2.SIFT_create()
        return sift.detectAndCompute(image, None)

    def match(self, descriptors_ref, descriptors_src):
        if descriptors_ref is None or descriptors_src is None:
            return []
        matcher = cv2.BFMatcher(cv2.NORM_L2)
        candidates = matcher.knnMatch(descriptors_src, descriptors_ref, k=2)
        return [
            match
            for match, second_match in candidates
            if match.distance < self.ratio_threshold * second_match.distance
        ]

    def find_homography(self, keypoints_ref, keypoints_src, matches):
        if len(matches) < 4:
            return None, None
        source_points = np.float32([keypoints_src[match.queryIdx].pt for match in matches])
        reference_points = np.float32([keypoints_ref[match.trainIdx].pt for match in matches])
        homography, mask = cv2.findHomography(
            source_points,
            reference_points,
            cv2.RANSAC,
            self.ransac_threshold,
        )
        return homography, mask.ravel().astype(bool) if mask is not None else None

    def save_images(
        self,
        reference,
        source,
        keypoints_ref,
        keypoints_src,
        matches,
        inlier_mask,
        homography,
    ):
        reference_keypoints = cv2.drawKeypoints(
            reference,
            keypoints_ref,
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        source_keypoints = cv2.drawKeypoints(
            source,
            keypoints_src,
            None,
            flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS,
        )
        activation = np.concatenate([reference_keypoints, source_keypoints], axis=1)
        cv2.imwrite(str(self.output_dir / "feature_activations.png"), activation)

        match_image = cv2.drawMatches(
            source,
            keypoints_src,
            reference,
            keypoints_ref,
            matches,
            None,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        cv2.imwrite(str(self.output_dir / "feature_matches.png"), match_image)

        inlier_matches = []
        if inlier_mask is not None:
            inlier_matches = [match for match, is_inlier in zip(matches, inlier_mask) if is_inlier]
        correlation_image = cv2.drawMatches(
            source,
            keypoints_src,
            reference,
            keypoints_ref,
            inlier_matches,
            None,
            matchColor=(0, 255, 255),
            singlePointColor=(255, 0, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        cv2.imwrite(str(self.output_dir / "feature_correlation.png"), correlation_image)

        if homography is None:
            warped_source = source.copy()
            reference_view = reference
            source_view = source
        else:
            source_height, source_width = source.shape[:2]
            source_corners = np.float32([
                [0, 0],
                [source_width - 1, 0],
                [source_width - 1, source_height - 1],
                [0, source_height - 1],
            ]).reshape(-1, 1, 2)
            transformed_corners = cv2.perspectiveTransform(source_corners, homography).reshape(-1, 2)
            reference_corners = np.float32([
                [0, 0],
                [reference.shape[1] - 1, 0],
                [reference.shape[1] - 1, reference.shape[0] - 1],
                [0, reference.shape[0] - 1],
            ])
            all_corners = np.concatenate([transformed_corners, reference_corners], axis=0)
            min_x, min_y = np.floor(all_corners.min(axis=0)).astype(int)
            max_x, max_y = np.ceil(all_corners.max(axis=0)).astype(int)
            translation = np.array([
                [1.0, 0.0, -min_x],
                [0.0, 1.0, -min_y],
                [0.0, 0.0, 1.0],
            ], dtype=np.float64)
            canvas_size = (max_x - min_x + 1, max_y - min_y + 1)
            reference_view = cv2.warpPerspective(
                reference,
                translation,
                canvas_size,
                borderMode=cv2.BORDER_REPLICATE,
            )
            warped_source = cv2.warpPerspective(
                source,
                translation @ homography,
                canvas_size,
                borderMode=cv2.BORDER_REPLICATE,
            )
            source_view = warped_source

        import matplotlib.pyplot as plt

        overlay = cv2.addWeighted(reference_view, 0.5, source_view, 0.5, 0)
        cv2.imwrite(str(self.output_dir / "reference.png"), reference_view)
        cv2.imwrite(str(self.output_dir / "source.png"), source)
        cv2.imwrite(str(self.output_dir / "registered_source.png"), source_view)
        cv2.imwrite(str(self.output_dir / "overlay.png"), overlay)
        figure, axes = plt.subplots(2, 2, figsize=(12, 10))
        panels = [
            (reference_view, "Reference"),
            (source, "Original source"),
            (source_view, "Registered source"),
            (overlay, "Overlay"),
        ]
        for axis, (image, title) in zip(axes.flat, panels):
            axis.imshow(image, cmap="gray", vmin=0, vmax=255)
            axis.set_title(title)
            axis.axis("off")
        figure.tight_layout()
        figure.savefig(self.output_dir / "homography_overlay.png", dpi=200, bbox_inches="tight")
        plt.close(figure)


Predict_sift = PredictSift
