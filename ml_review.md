# ML Implementation Review & Next Steps

## 1. Summary of ML Developer's Work
The ML developer has successfully built the core Image Registration & Prediction logic. Here is what has been accomplished:

*   **FastAPI Prototype (`aloo.py`)**: A functional prototype server with a built-in HTML frontend to demonstrate SIFT-based image registration, keypoint detection, homography, and overlay visualizations.
*   **Modular ML Architecture (`src/`)**: 
    *   **Services (`predict_sift.py`, `model_prediction.py`)**: Includes well-structured logic to predict homography matrices. It has support for traditional computer vision (SIFT/RANSAC) and deep learning models (PyTorch).
    *   **MLflow Integration**: The model prediction pipeline (`model_prediction.py` and `test.py`) is successfully wired up to pull trained models from an MLflow tracking server (`https://dagshub.com/vanshsharma7832/sih_isro.mlflow`).
*   **Visualizations**: Robust scripts to generate overlay images, feature correlations, and keypoint matching diagrams, which are saved locally in a `temp/` directory.

## 2. Review & Feedback for the ML Developer
*   **Great modularity**: The abstraction of `PredictSift` and `Model_prediction` inside the `src/services` folder makes it very easy to integrate into the final backend.
*   **Action item**: Ensure that `model_prediction.py` gracefully handles scenarios where MLflow is unreachable or times out.
*   **Action item**: Currently, the models save visualizations directly to the local filesystem (`temp/` directory). In a production environment, we should modify the services to return byte streams or upload these directly to cloud storage (like S3) to avoid stateful local storage on the API server.

---

## 3. Next Steps (Backend Developer Responsibilities)
As the backend developer, your job is to take this ML logic and turn it into a production-ready API.

1.  **Refactor the Prototype (`aloo.py`)**:
    *   Remove the hardcoded HTML template. The backend should only serve JSON responses, not HTML.
    *   Create clean RESTful endpoints (e.g., `POST /api/v1/register`).
2.  **Cloud Storage Integration**:
    *   Instead of saving images to `output_dir = "temp"`, integrate an AWS S3 client (or similar) to upload `ref_image`, `src_image`, and the resulting overlay images to the cloud. Return the public/presigned URLs in the JSON response.
3.  **Database Integration**:
    *   Set up a database (PostgreSQL/MongoDB) to track user requests, store the homography matrix data, and log processing times and image URLs.
4.  **Asynchronous Processing (Optional but recommended)**:
    *   If the deep learning model (`RegistrationNetwork.py`) takes a long time to run, move the inference out of the main request thread using **Celery** or **FastAPI BackgroundTasks** and implement a polling/webhook mechanism for the frontend.
5.  **Error Handling & Validation**:
    *   Add proper HTTP exceptions, file size limits, and MIME type validation for the uploaded images.

---

## 4. Handoff to Frontend Developer
Once the backend is refactored, you need to provide the following to the Frontend Developer:

*   **API Documentation**: Provide the FastAPI Swagger URL (usually `http://localhost:8000/docs`).
*   **Endpoints Details**:
    *   **Endpoint**: `POST /api/v1/register`
    *   **Request Type**: `multipart/form-data`
    *   **Payload**: `ref_image` (File), `src_image` (File)
*   **Expected JSON Response Format**:
    Give them the JSON schema they should expect. For example:
    ```json
    {
      "status": "success",
      "data": {
        "matches_found": 125,
        "inliers": 110,
        "homography_matrix": [[...], [...], [...]],
        "images": {
          "feature_activation_url": "https://s3.amazonaws.com/.../activation.png",
          "homography_overlay_url": "https://s3.amazonaws.com/.../overlay.png",
          "registered_source_url": "https://s3.amazonaws.com/.../registered.png"
        }
      }
    }
    ```
