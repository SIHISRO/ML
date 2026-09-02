import base64
import cv2
import numpy as np
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse

app = FastAPI()

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Sift-Based Image Registration & Visual Pipeline</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-color: #0b0f19;
            --card-bg: rgba(30, 41, 59, 0.7);
            --card-border: rgba(255, 255, 255, 0.08);
            --accent-cyan: #38bdf8;
            --accent-purple: #818cf8;
            --accent-green: #34d399;
            --accent-amber: #fbbf24;
            --text-main: #f8fafc;
            --text-muted: #94a3b8;
        }

        * {
            box-sizing: border-box;
        }

        body {
            font-family: 'Inter', system-ui, -apple-system, sans-serif;
            background-color: var(--bg-color);
            background-image: 
                radial-gradient(at 20% 20%, rgba(56, 189, 248, 0.08) 0px, transparent 50%),
                radial-gradient(at 80% 80%, rgba(129, 140, 248, 0.08) 0px, transparent 50%);
            color: var(--text-main);
            margin: 0;
            padding: 30px 15px;
            display: flex;
            flex-direction: column;
            align-items: center;
            min-height: 100vh;
        }

        .container {
            max-width: 1200px;
            width: 100%;
            background: var(--card-bg);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--card-border);
            padding: 35px;
            border-radius: 20px;
            box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.6);
        }

        .header {
            text-align: center;
            margin-bottom: 35px;
        }

        .header h1 {
            font-size: 2.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, var(--accent-cyan), var(--accent-purple));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin: 0 0 10px 0;
        }

        .header p {
            color: var(--text-muted);
            margin: 0;
            font-size: 1rem;
        }

        form {
            display: flex;
            flex-direction: column;
            gap: 20px;
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.05);
            padding: 25px;
            border-radius: 14px;
            margin-bottom: 30px;
        }

        .form-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
        }

        .input-group {
            display: flex;
            flex-direction: column;
            gap: 8px;
        }

        label {
            font-weight: 600;
            font-size: 0.9rem;
            color: var(--accent-cyan);
            letter-spacing: 0.5px;
        }

        input[type="file"] {
            background: rgba(30, 41, 59, 0.8);
            color: var(--text-main);
            padding: 12px;
            border-radius: 8px;
            border: 1px solid rgba(255, 255, 255, 0.1);
            outline: none;
            cursor: pointer;
            transition: border-color 0.2s;
        }

        input[type="file"]:hover {
            border-color: var(--accent-cyan);
        }

        button {
            background: linear-gradient(135deg, #0284c7, #4f46e5);
            color: white;
            border: none;
            padding: 14px 24px;
            font-size: 1rem;
            font-weight: 600;
            border-radius: 10px;
            cursor: pointer;
            transition: all 0.25s ease;
            box-shadow: 0 4px 14px rgba(2, 132, 199, 0.4);
        }

        button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(2, 132, 199, 0.6);
            background: linear-gradient(135deg, #0369a1, #4338ca);
        }

        .stats-bar {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-bottom: 35px;
        }

        .stat-card {
            background: rgba(15, 23, 42, 0.7);
            border: 1px solid var(--card-border);
            padding: 16px;
            border-radius: 12px;
            text-align: center;
        }

        .stat-value {
            font-size: 1.6rem;
            font-weight: 700;
            color: var(--accent-cyan);
        }

        .stat-label {
            font-size: 0.82rem;
            color: var(--text-muted);
            margin-top: 4px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .section-title {
            font-size: 1.3rem;
            font-weight: 600;
            color: var(--text-main);
            margin: 35px 0 15px 0;
            display: flex;
            align-items: center;
            gap: 10px;
        }

        .section-title .badge {
            background: rgba(56, 189, 248, 0.15);
            color: var(--accent-cyan);
            padding: 4px 10px;
            border-radius: 20px;
            font-size: 0.78rem;
            font-weight: 700;
        }

        .results-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
            gap: 22px;
        }

        .card {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid var(--card-border);
            padding: 18px;
            border-radius: 14px;
            text-align: center;
            transition: transform 0.2s, border-color 0.2s;
        }

        .card:hover {
            border-color: rgba(56, 189, 248, 0.3);
        }

        .card h3 {
            margin: 0 0 12px 0;
            font-size: 1.05rem;
            font-weight: 600;
            color: var(--accent-cyan);
        }

        .card img {
            max-width: 100%;
            height: auto;
            border-radius: 10px;
            box-shadow: 0 8px 16px rgba(0,0,0,0.4);
            border: 1px solid rgba(255,255,255,0.05);
        }

        .full-width-card {
            grid-column: 1 / -1;
        }

        .error {
            color: #f87171;
            background: rgba(127, 29, 29, 0.4);
            border: 1px solid #ef4444;
            padding: 16px;
            border-radius: 10px;
            margin-top: 25px;
            text-align: center;
            font-weight: 500;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>SIFT Image Registration & Visual Pipeline</h1>
            <p>Feature Detection • Keypoint Matching • RANSAC Homography • Visual Alignment</p>
        </div>
        
        <form action="/register" method="post" enctype="multipart/form-data">
            <div class="form-grid">
                <div class="input-group">
                    <label>Reference Image (Target)</label>
                    <input type="file" name="ref_image" accept="image/*" required>
                </div>
                <div class="input-group">
                    <label>Source Image (To be Aligned)</label>
                    <input type="file" name="src_image" accept="image/*" required>
                </div>
            </div>
            <button type="submit">⚡ Align & Visualize Registration Pipeline</button>
        </form>
        {content}
    </div>
</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
async def get_index():
    return HTML_TEMPLATE.replace("{content}", "")

@app.post("/register", response_class=HTMLResponse)
async def register(ref_image: UploadFile = File(...), src_image: UploadFile = File(...)):
    ref_bytes = await ref_image.read()
    src_bytes = await src_image.read()

    ref_np = np.frombuffer(ref_bytes, np.uint8)
    src_np = np.frombuffer(src_bytes, np.uint8)

    ref_img = cv2.imdecode(ref_np, cv2.IMREAD_COLOR)
    src_img = cv2.imdecode(src_np, cv2.IMREAD_COLOR)

    if ref_img is None or src_img is None:
        error_html = '<div class="error">Invalid image files uploaded.</div>'
        return HTML_TEMPLATE.replace("{content}", error_html)

    # Downscale images if they exceed maximum dimensions to prevent RAM ballooning
    MAX_DIM = 1200

    def resize_max_dim(img, max_dim=MAX_DIM):
        h, w = img.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            new_w = int(w * scale)
            new_h = int(h * scale)
            return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return img

    ref_img_proc = resize_max_dim(ref_img)
    src_img_proc = resize_max_dim(src_img)

    # 1. Feature Detection with SIFT
    sift = cv2.SIFT_create(nfeatures=2500)
    kp1, des1 = sift.detectAndCompute(ref_img_proc, None)
    kp2, des2 = sift.detectAndCompute(src_img_proc, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        error_html = '<div class="error">Could not detect enough SIFT features in the uploaded images.</div>'
        return HTML_TEMPLATE.replace("{content}", error_html)

    # 2. Keypoint Matching using FLANN
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des2, des1, k=2)

    good = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good.append(m)

    if len(good) < 4:
        error_html = '<div class="error">Not enough matching feature points found between the images.</div>'
        return HTML_TEMPLATE.replace("{content}", error_html)

    # 3. Compute Homography Matrix using RANSAC
    src_pts = np.float32([kp2[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp1[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is None or mask is None:
        error_html = '<div class="error">Failed to compute transformation matrix (Homography).</div>'
        return HTML_TEMPLATE.replace("{content}", error_html)

    inliers_count = int(np.sum(mask))
    inlier_matches = [good[i] for i in range(len(good)) if mask[i] == 1]

    # 4. Warp Perspective Transformation
    h, w = ref_img_proc.shape[:2]
    registered_img = cv2.warpPerspective(src_img_proc, H, (w, h))

    # --- GENERATE VISUALIZATIONS ---

    # Visual 1: Draw Keypoints on Ref & Src Images
    ref_kp_img = cv2.drawKeypoints(
        ref_img_proc, kp1, None, 
        color=(56, 189, 248), 
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )
    src_kp_img = cv2.drawKeypoints(
        src_img_proc, kp2, None, 
        color=(251, 191, 36), 
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Visual 2: Draw Correspondence Matching Lines (Top 40 Inliers)
    matches_img = cv2.drawMatches(
        src_img_proc, kp2, 
        ref_img_proc, kp1, 
        inlier_matches[:40], None, 
        matchColor=(52, 211, 153), 
        singlePointColor=(148, 163, 184), 
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )

    # Visual 3: Blended Overlay (50% Ref + 50% Registered)
    blend_img = cv2.addWeighted(ref_img_proc, 0.5, registered_img, 0.5, 0)

    # Helper function for Base64 JPEG encoding
    def to_b64(img, quality=80):
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        _, encoded = cv2.imencode('.jpg', img, encode_param)
        return base64.b64encode(encoded).decode('utf-8')

    ref_b64 = to_b64(ref_img_proc)
    src_b64 = to_b64(src_img_proc)
    ref_kp_b64 = to_b64(ref_kp_img)
    src_kp_b64 = to_b64(src_kp_img)
    matches_b64 = to_b64(matches_img)
    reg_b64 = to_b64(registered_img)
    blend_b64 = to_b64(blend_img)

    inlier_ratio = round((inliers_count / len(good)) * 100, 1)

    results_html = f'''
    <!-- Statistics Summary Bar -->
    <div class="stats-bar">
        <div class="stat-card">
            <div class="stat-value">{len(kp1)}</div>
            <div class="stat-label">Ref Keypoints</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len(kp2)}</div>
            <div class="stat-label">Source Keypoints</div>
        </div>
        <div class="stat-card">
            <div class="stat-value">{len(good)}</div>
            <div class="stat-label">Raw Matches</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color: var(--accent-green);">{inliers_count}</div>
            <div class="stat-label">RANSAC Inliers</div>
        </div>
        <div class="stat-card">
            <div class="stat-value" style="color: var(--accent-amber);">{inlier_ratio}%</div>
            <div class="stat-label">Match Precision</div>
        </div>
    </div>

    <!-- Step 1: Input Images & Detected Keypoints -->
    <div class="section-title">
        <span>Step 1: SIFT Feature & Keypoint Detection</span>
        <span class="badge">Feature Extraction</span>
    </div>
    <div class="results-grid">
        <div class="card">
            <h3>Reference Image Keypoints ({len(kp1)})</h3>
            <img src="data:image/jpeg;base64,{ref_kp_b64}" alt="Reference Keypoints" />
        </div>
        <div class="card">
            <h3>Source Image Keypoints ({len(kp2)})</h3>
            <img src="data:image/jpeg;base64,{src_kp_b64}" alt="Source Keypoints" />
        </div>
    </div>

    <!-- Step 2: Feature Matching & Line Connections -->
    <div class="section-title">
        <span>Step 2: Keypoint Matching Lines (Source ➔ Reference)</span>
        <span class="badge">Inlier Correspondences</span>
    </div>
    <div class="results-grid">
        <div class="card full-width-card">
            <h3>Keypoint Match Lines (Top RANSAC Inliers Drawn)</h3>
            <img src="data:image/jpeg;base64,{matches_b64}" alt="Feature Matching Lines" />
        </div>
    </div>

    <!-- Step 3: Transformation & Final Registered Results -->
    <div class="section-title">
        <span>Step 3: Homography Transformation & Registered Alignment</span>
        <span class="badge">Final Output</span>
    </div>
    <div class="results-grid">
        <div class="card">
            <h3>Original Source Image</h3>
            <img src="data:image/jpeg;base64,{src_b64}" alt="Source Image" />
        </div>
        <div class="card">
            <h3>Warped / Registered Result</h3>
            <img src="data:image/jpeg;base64,{reg_b64}" alt="Registered Image" />
        </div>
        <div class="card">
            <h3>50/50 Overlay Verification (Ref + Registered)</h3>
            <img src="data:image/jpeg;base64,{blend_b64}" alt="Blended Overlay" />
        </div>
    </div>
    '''
    return HTML_TEMPLATE.replace("{content}", results_html)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)