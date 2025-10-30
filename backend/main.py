# from fastapi import FastAPI, File, UploadFile
# import uuid, os, shutil
# from ultralytics import YOLO

# app = FastAPI()

# UPLOAD_DIR = "uploads"
# RESULT_DIR = "results"
# MODEL_DIR = "models"

# # Load models once (saves time)
# human_model = YOLO(os.path.join(MODEL_DIR, "human.pt"))
# animal_vehicle_model = YOLO(os.path.join(MODEL_DIR, "vehicle_animal.pt"))
# forest_model = YOLO(os.path.join(MODEL_DIR, "forest.pt"))

# os.makedirs(UPLOAD_DIR, exist_ok=True)
# os.makedirs(RESULT_DIR, exist_ok=True)

# @app.post("/upload_video/")
# async def upload_video(file: UploadFile = File(...)):
#     # Generate UUID for session
#     session_id = str(uuid.uuid4())
#     upload_path = os.path.join(UPLOAD_DIR, session_id)
#     result_path = os.path.join(RESULT_DIR, session_id)
#     os.makedirs(upload_path, exist_ok=True)
#     os.makedirs(result_path, exist_ok=True)

#     # Save uploaded file
#     file_path = os.path.join(upload_path, file.filename)
#     with open(file_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     # Run YOLO detections
#     # (You can change predict() to video inference too)
#     human_model.predict(source=file_path, save=True, project=result_path, name="human")
#     animal_vehicle_model.predict(source=file_path, save=True, project=result_path, name="vehicle_animal")
#     forest_model.predict(source=file_path, save=True, project=result_path, name="forest")

#     return {
#         "uuid": session_id,
#         "filename": file.filename,
#         "message": "Processing complete!",
#         "results_folder": result_path,
#     }


# detect
# from fastapi import FastAPI, File, UploadFile
# import uuid, os, shutil
# from ultralytics import YOLO
# import cv2
# import numpy as np

# app = FastAPI()

# UPLOAD_DIR = "uploads"
# RESULT_DIR = "results"
# MODEL_DIR = "models"

# # Load models
# human_model = YOLO(os.path.join(MODEL_DIR, "human.pt"))
# animal_vehicle_model = YOLO(os.path.join(MODEL_DIR, "vehicle_animal.pt"))
# forest_model = YOLO(os.path.join(MODEL_DIR, "forest.pt"))

# os.makedirs(UPLOAD_DIR, exist_ok=True)
# os.makedirs(RESULT_DIR, exist_ok=True)

# @app.post("/upload_video/")
# async def upload_video(file: UploadFile = File(...)):
#     session_id = str(uuid.uuid4())
#     upload_path = os.path.join(UPLOAD_DIR, session_id)
#     result_path = os.path.join(RESULT_DIR, session_id)
#     os.makedirs(upload_path, exist_ok=True)
#     os.makedirs(result_path, exist_ok=True)

#     file_path = os.path.join(upload_path, file.filename)
#     with open(file_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     # Run detections
#     res_human = human_model.predict(source=file_path, save=False)
#     res_animal_vehicle = animal_vehicle_model.predict(source=file_path, save=False)
#     res_forest = forest_model.predict(source=file_path, save=False)

#     # 🧠 Get max confidence for humans
#     human_conf = max([float(b.conf[0]) for b in res_human[0].boxes], default=0)

#     # 🧠 Filter only 'animal' class (class 2)
#     animal_boxes = []
#     for box in res_animal_vehicle[0].boxes:
#         cls = int(box.cls[0])
#         if cls == 2:  # class 2 = 'animal'
#             animal_boxes.append(box)

#     animal_conf = max([float(b.conf[0]) for b in animal_boxes], default=0)

#     # 🧩 Decision: pick the one with higher confidence
#     if human_conf >= animal_conf:
#         animal_boxes = []  # ignore animals
#     else:
#         res_human[0].boxes = []  # ignore humans

#     img = cv2.imread(file_path)
#     colors = {
#         "human": (0, 255, 0),
#         "animal": (255, 0, 0),
#         "forest": (0, 0, 255)
#     }

#     # ✅ Draw human boxes (if any)
#     for box in res_human[0].boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         conf = float(box.conf[0])
#         cv2.rectangle(img, (x1, y1), (x2, y2), colors["human"], 2)
#         cv2.putText(img, f"Human {conf:.2f}", (x1, y1 - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors["human"], 2)

#     # ✅ Draw animal boxes (if any)
#     for box in animal_boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         conf = float(box.conf[0])
#         cv2.rectangle(img, (x1, y1), (x2, y2), colors["animal"], 2)
#         cv2.putText(img, f"Animal {conf:.2f}", (x1, y1 - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors["animal"], 2)

#     # ✅ Draw forest/deforestation (from forest model)
#     for box in res_forest[0].boxes:
#         x1, y1, x2, y2 = map(int, box.xyxy[0])
#         conf = float(box.conf[0])
#         cv2.rectangle(img, (x1, y1), (x2, y2), colors["forest"], 2)
#         cv2.putText(img, f"Deforestation {conf:.2f}", (x1, y1 - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors["forest"], 2)

#     # Save final image
#     combined_output = os.path.join(result_path, "final_combined.jpg")
#     cv2.imwrite(combined_output, img)

#     return {
#         "uuid": session_id,
#         "filename": file.filename,
#         "message": "✅ Combined detection complete!",
#         "results_folder": result_path,
#         "final_output": combined_output
#     }


# # eppah dei
# from fastapi import FastAPI, File, UploadFile, HTTPException
# import uuid
# import os
# import shutil
# from ultralytics import YOLO
# import cv2
# import numpy as np

# app = FastAPI()

# UPLOAD_DIR = "uploads"
# RESULT_DIR = "results"
# MODEL_DIR = "models"

# # Load trained models
# human_model = YOLO(os.path.join(MODEL_DIR, "human.pt"))
# animal_vehicle_model = YOLO(os.path.join(MODEL_DIR, "vehicle_animal.pt"))
# forest_model = YOLO(os.path.join(MODEL_DIR, "forest.pt"))  # segmentation model

# os.makedirs(UPLOAD_DIR, exist_ok=True)
# os.makedirs(RESULT_DIR, exist_ok=True)


# def process_frame(img, session_path):
#     """Run detection + segmentation + label correction"""

#     # Run models
#     res_vehicle_animal = animal_vehicle_model.predict(source=img, save=False, verbose=False)
#     res_human = human_model.predict(source=img, save=False, verbose=False)
#     res_forest = forest_model.predict(source=img, save=False, verbose=False)

#     boxes_vehicle_animal = res_vehicle_animal[0].boxes
#     boxes_human = res_human[0].boxes
#     final_boxes = []

#     # ---- Step 1: Take only classes 2 (animal) and 4 (vehicle) ----
#     for b in boxes_vehicle_animal:
#         cls = int(b.cls[0])
#         if cls not in [2, 4]:
#             continue  # skip forest/deforested/human
#         x1, y1, x2, y2 = map(int, b.xyxy[0])
#         conf = float(b.conf[0])
#         label = "Animal" if cls == 2 else "Vehicle"
#         final_boxes.append({"bbox": (x1, y1, x2, y2), "conf": conf, "label": label})

#     # ---- Step 2: Compare with human detections ----
#     for h in boxes_human:
#         hx1, hy1, hx2, hy2 = map(int, h.xyxy[0])
#         h_conf = float(h.conf[0])
#         h_center = ((hx1 + hx2) / 2, (hy1 + hy2) / 2)

#         replaced = False
#         for fb in final_boxes:
#             x1, y1, x2, y2 = fb["bbox"]
#             if x1 < h_center[0] < x2 and y1 < h_center[1] < y2:
#                 # overlapping area found
#                 if h_conf > fb["conf"]:
#                     fb["label"] = "Human"
#                     fb["conf"] = h_conf
#                     replaced = True
#                 break

#         if not replaced:
#             final_boxes.append({"bbox": (hx1, hy1, hx2, hy2), "conf": h_conf, "label": "Human"})

#     # ---- Step 3: Draw detections ----
#     color_map = {
#         "Human": (0, 255, 0),
#         "Animal": (255, 0, 0),
#         "Vehicle": (255, 255, 0)
#     }

#     for box in final_boxes:
#         (x1, y1, x2, y2) = box["bbox"]
#         conf = box["conf"]
#         label = box["label"]
#         color = color_map[label]
#         cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
#         cv2.putText(img, f"{label} {conf:.2f}", (x1, y1 - 5),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

#     # ---- Step 4: Forest segmentation (only deforested) ----
#      # ---- Step 4: Forest segmentation (tiling for better accuracy) ----
#     tile_size = 512
#     overlap = 64  # small overlap to avoid border artifacts
#     h, w, _ = img.shape
#     combined_mask = np.zeros((h, w), dtype=np.float32)

#     for y in range(0, h, tile_size - overlap):
#         for x in range(0, w, tile_size - overlap):
#             x_end = min(x + tile_size, w)
#             y_end = min(y + tile_size, h)
#             tile = img[y:y_end, x:x_end]

#             res_tile = forest_model.predict(source=tile, save=False, verbose=False)
#             if hasattr(res_tile[0], "masks") and res_tile[0].masks is not None:
#                 for i, m in enumerate(res_tile[0].masks.data):
#                     mask = m.cpu().numpy()
#                     mask = cv2.resize(mask, (x_end - x, y_end - y))

#                     cls_idx = int(res_tile[0].boxes.cls[i]) if len(res_tile[0].boxes) > i else 0
#                     if cls_idx == 0:  # if 'deforested_area' in your YAML is class 1
#                         combined_mask[y:y_end, x:x_end] = np.maximum(
#                             combined_mask[y:y_end, x:x_end], mask
#                         )

#     # apply red overlay for deforested regions
#     if np.any(combined_mask > 0):
#         red_overlay = np.zeros_like(img, dtype=np.uint8)
#         red_overlay[:, :, 2] = 255
#         red_overlay = (red_overlay * combined_mask[..., None]).astype(np.uint8)
#         img = cv2.addWeighted(img, 1.0, red_overlay, 0.35, 0)

#     return img


# @app.post("/upload_video/")
# async def upload_file(file: UploadFile = File(...)):
#     session_id = str(uuid.uuid4())
#     upload_path = os.path.join(UPLOAD_DIR, session_id)
#     result_path = os.path.join(RESULT_DIR, session_id)
#     os.makedirs(upload_path, exist_ok=True)
#     os.makedirs(result_path, exist_ok=True)

#     file_path = os.path.join(upload_path, file.filename)
#     with open(file_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     ext = file.filename.lower().split('.')[-1]

#     if ext in ["mp4", "mov", "avi", "mkv"]:
#         # --- Video Processing ---
#         cap = cv2.VideoCapture(file_path)
#         if not cap.isOpened():
#             raise HTTPException(status_code=400, detail="Could not open video file")

#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#         out_path = os.path.join(result_path, "final_combined.mp4")
#         out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             processed = process_frame(frame, result_path)
#             out.write(processed)

#         cap.release()
#         out.release()
#         combined_output = out_path

#     else:
#         # --- Image Processing ---
#         img = cv2.imread(file_path)
#         if img is None:
#             raise HTTPException(status_code=400, detail="Uploaded image could not be read or is corrupted")

#         processed = process_frame(img, result_path)
#         combined_output = os.path.join(result_path, "final_combined.jpg")
#         cv2.imwrite(combined_output, processed)

#     return {
#         "uuid": session_id,
#         "filename": file.filename,
#         "message": "✅ Combined detection + deforestation segmentation complete!",
#         "results_folder": result_path,
#         "final_output": combined_output
#     }

# reallygreatcode
from fastapi import FastAPI, File, UploadFile, HTTPException
import uuid
import os
import shutil
from ultralytics import YOLO
import cv2
import numpy as np

app = FastAPI()

UPLOAD_DIR = "uploads"
RESULT_DIR = "results"
MODEL_DIR = "models"

# Load YOLO models
human_model = YOLO(os.path.join(MODEL_DIR, "human.pt"))
animal_vehicle_model = YOLO(os.path.join(MODEL_DIR, "vehicle_animal.pt"))
forest_model = YOLO(os.path.join(MODEL_DIR, "forest.pt"))  # segmentation model

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(RESULT_DIR, exist_ok=True)


def forest_segmentation_only(img):
    """Return image with forest segmentation + deforestation highlight (no humans/animals/vehicles)."""
    tile_size = 512
    overlap = 64
    h, w, _ = img.shape
    forest_mask = np.zeros((h, w), dtype=np.float32)
    deforest_mask = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h, tile_size - overlap):
        for x in range(0, w, tile_size - overlap):
            x_end = min(x + tile_size, w)
            y_end = min(y + tile_size, h)
            tile = img[y:y_end, x:x_end]

            res_tile = forest_model.predict(source=tile, save=False, verbose=False)
            if hasattr(res_tile[0], "masks") and res_tile[0].masks is not None:
                for i, m in enumerate(res_tile[0].masks.data):
                    mask = m.cpu().numpy()
                    mask = cv2.resize(mask, (x_end - x, y_end - y))
                    cls_idx = int(res_tile[0].boxes.cls[i]) if len(res_tile[0].boxes) > i else 0

                    if cls_idx == 0:  # deforested
                        deforest_mask[y:y_end, x:x_end] = np.maximum(deforest_mask[y:y_end, x:x_end], mask)
                    else:  # forest
                        forest_mask[y:y_end, x:x_end] = np.maximum(forest_mask[y:y_end, x:x_end], mask)

    masked_img = (img * forest_mask[..., None]).astype(np.uint8)
    background_dim = (img * (1 - forest_mask[..., None]) * 0.4).astype(np.uint8)
    combined_forest = cv2.add(masked_img, background_dim)

    if np.any(deforest_mask > 0):
        red_overlay = np.zeros_like(img, dtype=np.uint8)
        red_overlay[:, :, 2] = 255
        red_overlay = (red_overlay * deforest_mask[..., None]).astype(np.uint8)
        combined_forest = cv2.addWeighted(combined_forest, 0.7, red_overlay, 0.3, 0)

    return combined_forest
def detection_only(img):
    """Return image with human, animal, and vehicle detection (no forest mask)."""
    res_vehicle_animal = animal_vehicle_model.predict(source=img, save=False, verbose=False)
    res_human = human_model.predict(source=img, save=False, verbose=False)

    boxes_vehicle_animal = res_vehicle_animal[0].boxes
    boxes_human = res_human[0].boxes

    output_img = img.copy()
    color_map = {
        "Human": (0, 255, 0),
        "Animal": (255, 0, 0),
        "Vehicle": (255, 255, 0),
        "Living Thing": (0, 165, 255)
    }

    # Sum confidence for humans and animals
    human_conf_sum = sum(float(h.conf[0]) for h in boxes_human)
    animal_conf_sum = sum(float(b.conf[0]) for b in boxes_vehicle_animal if int(b.cls[0]) == 2)

    conf_threshold_general = 0.6  # High threshold to reduce false positives
    conf_threshold_human = 0.3

    # Vehicles - always labeled Vehicle, only draw if confidence above threshold
    for b in boxes_vehicle_animal:
        cls = int(b.cls[0])
        conf = float(b.conf[0])
        if cls == 4 and conf >= conf_threshold_general:  # Vehicle
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            label = "Vehicle"
            color = color_map[label]
            cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(output_img, f"{label} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Animals - labeled Animal if animal_conf >= human_conf else Living Thing, confidence filtered
    for b in boxes_vehicle_animal:
        cls = int(b.cls[0])
        conf = float(b.conf[0])
        if cls == 2 and conf >= conf_threshold_general:
            x1, y1, x2, y2 = map(int, b.xyxy[0])
            label = "Animal" if animal_conf_sum >= human_conf_sum else "Living Thing"
            color = color_map[label]
            cv2.rectangle(output_img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(output_img, f"{label} {conf:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Humans - confidence, size, and aspect ratio filtering for strict validation
    for h in boxes_human:
        conf = float(h.conf[0])
        if conf < conf_threshold_human:
            continue
        hx1, hy1, hx2, hy2 = map(int, h.xyxy[0])
        width = hx2 - hx1
        height = hy2 - hy1

        if width < 30 or height < 60:
            continue  # too small to be human

        aspect_ratio = height / (width + 1e-6)
        if aspect_ratio < 1.3:  # humans usually taller than wide, stricter
            continue

        label = "Human"
        color = color_map[label]
        cv2.rectangle(output_img, (hx1, hy1), (hx2, hy2), color, 2)
        cv2.putText(output_img, f"{label} {conf:.2f}", (hx1, hy1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return output_img


@app.post("/upload_video/")
async def upload_file(file: UploadFile = File(...)):
    """Handles both image and video upload and returns two outputs:
       1. forest segmentation
       2. human+animal+vehicle detection
    """
    session_id = str(uuid.uuid4())
    upload_path = os.path.join(UPLOAD_DIR, session_id)
    result_path = os.path.join(RESULT_DIR, session_id)
    os.makedirs(upload_path, exist_ok=True)
    os.makedirs(result_path, exist_ok=True)

    file_path = os.path.join(upload_path, file.filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    ext = file.filename.lower().split('.')[-1]

    if ext in ["mp4", "mov", "avi", "mkv"]:
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise HTTPException(status_code=400, detail="Could not open video file")

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_FPS) or 30.0
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')

        forest_out_path = os.path.join(result_path, "forest_segmentation.mp4")
        detect_out_path = os.path.join(result_path, "detections.mp4")

        forest_out = cv2.VideoWriter(forest_out_path, fourcc, fps, (width, height))
        detect_out = cv2.VideoWriter(detect_out_path, fourcc, fps, (width, height))

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            forest_frame = forest_segmentation_only(frame)
            detect_frame = detection_only(frame)

            forest_out.write(forest_frame)
            detect_out.write(detect_frame)

        cap.release()
        forest_out.release()
        detect_out.release()

        outputs = {
            "forest_segmentation": forest_out_path,
            "detections": detect_out_path
        }

    else:
        img = cv2.imread(file_path)
        if img is None:
            raise HTTPException(status_code=400, detail="Uploaded image is invalid or corrupted")

        forest_img = forest_segmentation_only(img)
        detect_img = detection_only(img)

        forest_path = os.path.join(result_path, "forest_segmentation.jpg")
        detect_path = os.path.join(result_path, "detections.jpg")

        cv2.imwrite(forest_path, forest_img)
        cv2.imwrite(detect_path, detect_img)

        outputs = {
            "forest_segmentation": forest_path,
            "detections": detect_path
        }

    return {
        "uuid": session_id,
        "filename": file.filename,
        "message": "✅ Both outputs generated successfully!",
        "results_folder": result_path,
        "outputs": outputs
    }
# from fastapi import FastAPI, File, UploadFile, HTTPException
# import uuid
# import os
# import shutil
# from ultralytics import YOLO
# import cv2
# import numpy as np
# from collections import defaultdict
# import math


# app = FastAPI()


# UPLOAD_DIR = "uploads"
# RESULT_DIR = "results"
# MODEL_DIR = "models"


# # Load YOLO models
# human_model = YOLO(os.path.join(MODEL_DIR, "human.pt"))
# animal_vehicle_model = YOLO(os.path.join(MODEL_DIR, "vehicle_animal.pt"))
# forest_model = YOLO(os.path.join(MODEL_DIR, "forest.pt"))


# os.makedirs(UPLOAD_DIR, exist_ok=True)
# os.makedirs(RESULT_DIR, exist_ok=True)


# def calculate_iou(box1, box2):
#     x1_min, y1_min, x1_max, y1_max = box1
#     x2_min, y2_min, x2_max, y2_max = box2
#     inter_xmin = max(x1_min, x2_min)
#     inter_ymin = max(y1_min, y2_min)
#     inter_xmax = min(x1_max, x2_max)
#     inter_ymax = min(y1_max, y2_max)

#     if inter_xmax < inter_xmin or inter_ymax < inter_ymin:
#         return 0.0

#     inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
#     box1_area = (x1_max - x1_min) * (y1_max - y1_min)
#     box2_area = (x2_max - x2_min) * (y2_max - y2_min)
#     union_area = box1_area + box2_area - inter_area

#     return inter_area / union_area if union_area > 0 else 0.0



# def smart_augmentations(img):
#     """Reduced to 6 key augmentations for speed."""
#     h, w = img.shape[:2]
#     augmented = []

#     augmented.append(("original", img.copy(), lambda b: b))

#     # Horizontal flip
#     h_flip = cv2.flip(img, 1)
#     augmented.append(("h_flip", h_flip, lambda boxes: [
#         {**b, "box": (w - b["box"][2], b["box"][1], w - b["box"][0], b["box"][3])}
#         for b in boxes
#     ]))

#     # Brightness increase
#     bright = cv2.convertScaleAbs(img, alpha=1.3, beta=20)
#     augmented.append(("bright", bright, lambda b: b))

#     # Darker version
#     dark = cv2.convertScaleAbs(img, alpha=0.7, beta=-20)
#     augmented.append(("dark", dark, lambda b: b))

#     # Sharpen
#     kernel_sharp = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
#     sharpened = cv2.filter2D(img, -1, kernel_sharp)
#     augmented.append(("sharp", sharpened, lambda b: b))

#     # Vertical flip
#     v_flip = cv2.flip(img, 0)
#     augmented.append(("v_flip", v_flip, lambda boxes: [
#         {**b, "box": (b["box"][0], h - b["box"][3], b["box"][2], h - b["box"][1])}
#         for b in boxes
#     ]))

#     return augmented



# def smart_multi_scale_detection(img, model, cls_filter=None):
#     """Reduced to 5 scales and fewer augmentations for faster execution."""
#     all_detections = []
#     h, w = img.shape[:2]

#     scales = [0.75, 0.9, 1.0, 1.15, 1.3]

#     for scale in scales:
#         new_w, new_h = int(w * scale), int(h * scale)

#         if new_w < 256 or new_h < 256 or new_w > 5000 or new_h > 5000:
#             continue

#         scaled_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

#         augmented_images = smart_augmentations(scaled_img)

#         for aug_name, aug_img, reverse_fn in augmented_images:
#             results = model.predict(
#                 source=aug_img,
#                 save=False,
#                 verbose=False,
#                 conf=0.35,  # Increased confidence threshold
#                 iou=0.45,   # Increased IOU threshold for better NMS
#                 imgsz=640
#             )

#             boxes = results[0].boxes

#             for b in boxes:
#                 cls = int(b.cls[0])

#                 if cls_filter and cls not in cls_filter:
#                     continue

#                 x1, y1, x2, y2 = map(float, b.xyxy[0])
#                 conf = float(b.conf[0])

#                 x1, x2 = x1 / scale, x2 / scale
#                 y1, y2 = y1 / scale, y2 / scale

#                 detection = {
#                     "box": (int(x1), int(y1), int(x2), int(y2)),
#                     "conf": conf,
#                     "cls": cls,
#                     "scale": scale,
#                     "aug": aug_name
#                 }

#                 detection = reverse_fn([detection])[0]
#                 all_detections.append(detection)

#     return all_detections



# def soft_nms(detections, iou_threshold=0.4, sigma=0.5, score_threshold=0.25):
#     if not detections:
#         return []

#     detections = sorted(detections, key=lambda x: x["conf"], reverse=True)

#     for i in range(len(detections)):
#         if detections[i]["conf"] < score_threshold:
#             continue

#         for j in range(i + 1, len(detections)):
#             if detections[i]["cls"] != detections[j]["cls"]:
#                 continue

#             iou = calculate_iou(detections[i]["box"], detections[j]["box"])

#             if iou > iou_threshold:
#                 detections[j]["conf"] *= math.exp(-(iou * iou) / sigma)

#     return [d for d in detections if d["conf"] >= score_threshold]



# def cluster_based_fusion(detections, iou_threshold=0.3, min_cluster_size=3):
#     if not detections:
#         return []

#     class_groups = defaultdict(list)
#     for det in detections:
#         class_groups[det["cls"]].append(det)

#     fused_results = []

#     for cls, cls_dets in class_groups.items():
#         clusters = []
#         used = set()

#         for i, det1 in enumerate(cls_dets):
#             if i in used:
#                 continue

#             cluster = [det1]
#             used.add(i)

#             for j in range(i + 1, len(cls_dets)):
#                 if j in used:
#                     continue

#                 for cluster_det in cluster:
#                     iou = calculate_iou(det1["box"], cls_dets[j]["box"])
#                     if iou > iou_threshold:
#                         cluster.append(cls_dets[j])
#                         used.add(j)
#                         break

#             clusters.append(cluster)

#         for cluster in clusters:
#             if len(cluster) < min_cluster_size:
#                 if cluster:
#                     best = max(cluster, key=lambda x: x["conf"])
#                     if best["conf"] > 0.35:
#                         fused_results.append(best)
#                 continue

#             total_weight = sum(d["conf"] for d in cluster)

#             if total_weight == 0:
#                 continue

#             x1 = sum(d["box"][0] * d["conf"] for d in cluster) / total_weight
#             y1 = sum(d["box"][1] * d["conf"] for d in cluster) / total_weight
#             x2 = sum(d["box"][2] * d["conf"] for d in cluster) / total_weight
#             y2 = sum(d["box"][3] * d["conf"] for d in cluster) / total_weight

#             avg_conf = total_weight / len(cluster)
#             max_conf = max(d["conf"] for d in cluster)
#             voting_bonus = min(len(cluster) / 50.0, 0.2)

#             final_conf = min((avg_conf + max_conf) / 2 + voting_bonus, 1.0)

#             fused_results.append({
#                 "box": (int(x1), int(y1), int(x2), int(y2)),
#                 "conf": final_conf,
#                 "cls": cls,
#                 "votes": len(cluster)
#             })

#     return fused_results



# def smart_forest_segmentation(img):
#     tile_sizes = [512, 640]  # Reduced tile sizes for speed
#     overlap = 80
#     h, w, _ = img.shape

#     forest_masks = []
#     deforest_masks = []

#     for img_scale in [0.9, 1.0, 1.15]:  # Reduced scales
#         scaled_h, scaled_w = int(h * img_scale), int(w * img_scale)

#         if scaled_h < 320 or scaled_w < 320:
#             continue

#         scaled_img = cv2.resize(img, (scaled_w, scaled_h), interpolation=cv2.INTER_CUBIC)

#         for tile_size in tile_sizes:
#             forest_mask = np.zeros((scaled_h, scaled_w), dtype=np.float32)
#             deforest_mask = np.zeros((scaled_h, scaled_w), dtype=np.float32)

#             for y in range(0, scaled_h, tile_size - overlap):
#                 for x in range(0, scaled_w, tile_size - overlap):
#                     x_end = min(x + tile_size, scaled_w)
#                     y_end = min(y + tile_size, scaled_h)
#                     tile = scaled_img[y:y_end, x:x_end]

#                     res_tile = forest_model.predict(
#                         source=tile,
#                         save=False,
#                         verbose=False,
#                         conf=0.20,
#                         iou=0.4
#                     )

#                     if hasattr(res_tile[0], "masks") and res_tile[0].masks is not None:
#                         for i, m in enumerate(res_tile[0].masks.data):
#                             mask = m.cpu().numpy()
#                             mask = cv2.resize(mask, (x_end - x, y_end - y))

#                             cls_idx = int(res_tile[0].boxes.cls[i]) if len(res_tile[0].boxes) > i else 0
#                             conf_val = float(res_tile[0].boxes.conf[i]) if len(res_tile[0].boxes) > i else 0.5

#                             mask = mask * conf_val

#                             if cls_idx == 0:
#                                 deforest_mask[y:y_end, x:x_end] = np.maximum(
#                                     deforest_mask[y:y_end, x:x_end], mask
#                                 )
#                             else:
#                                 forest_mask[y:y_end, x:x_end] = np.maximum(
#                                     forest_mask[y:y_end, x:x_end], mask
#                                 )

#             forest_mask = cv2.resize(forest_mask, (w, h))
#             deforest_mask = cv2.resize(deforest_mask, (w, h))

#             forest_masks.append(forest_mask)
#             deforest_masks.append(deforest_mask)

#     forest_masks = np.array(forest_masks)
#     deforest_masks = np.array(deforest_masks)

#     final_forest = np.median(forest_masks, axis=0)
#     final_deforest = np.median(deforest_masks, axis=0)

#     kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
#     kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
#     kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

#     final_forest = cv2.morphologyEx(final_forest, cv2.MORPH_OPEN, kernel_small)
#     final_forest = cv2.morphologyEx(final_forest, cv2.MORPH_CLOSE, kernel_large)

#     final_deforest = cv2.morphologyEx(final_deforest, cv2.MORPH_OPEN, kernel_small)
#     final_deforest = cv2.morphologyEx(final_deforest, cv2.MORPH_CLOSE, kernel_medium)

#     final_forest = cv2.GaussianBlur(final_forest, (5, 5), 1)
#     final_deforest = cv2.GaussianBlur(final_deforest, (5, 5), 1)

#     masked_img = (img * final_forest[..., None]).astype(np.uint8)
#     background_dim = (img * (1 - final_forest[..., None]) * 0.3).astype(np.uint8)
#     combined_forest = cv2.add(masked_img, background_dim)

#     if np.any(final_deforest > 0.3):
#         red_overlay = np.zeros_like(img, dtype=np.uint8)
#         red_overlay[:, :, 2] = 255
#         deforest_alpha = np.clip(final_deforest, 0, 1)[..., None]
#         red_overlay = (red_overlay * deforest_alpha).astype(np.uint8)
#         combined_forest = cv2.addWeighted(combined_forest, 0.65, red_overlay, 0.35, 0)

#     return combined_forest



# def smart_max_detection(img):
#     output_img = img.copy()
#     color_map = {"Human": (0, 255, 0), "Animal": (255, 0, 0), "Vehicle": (255, 255, 0)}

#     animal_vehicle_detections = smart_multi_scale_detection(
#         img, animal_vehicle_model, cls_filter=[2, 4]
#     )

#     human_detections = smart_multi_scale_detection(
#         img, human_model, cls_filter=None
#     )

#     all_detections = []

#     for det in animal_vehicle_detections:
#         det["label"] = "Animal" if det["cls"] == 2 else "Vehicle"
#         all_detections.append(det)

#     for det in human_detections:
#         det["label"] = "Human"
#         all_detections.append(det)

#     soft_nms_results = soft_nms(all_detections, iou_threshold=0.4, sigma=0.5, score_threshold=0.25)
#     fused = cluster_based_fusion(soft_nms_results, iou_threshold=0.3, min_cluster_size=3)

#     for det in fused:
#         if "label" not in det:
#             if det["cls"] == 2:
#                 det["label"] = "Animal"
#             elif det["cls"] == 4:
#                 det["label"] = "Vehicle"
#             else:
#                 det["label"] = "Human"

#     # Compare total confidence of humans vs. animals
#     total_human_conf = sum(det["conf"] for det in fused if det["label"] == "Human")
#     total_animal_conf = sum(det["conf"] for det in fused if det["label"] == "Animal")

#     if total_human_conf > total_animal_conf:
#         # Remove animal detections if humans dominate by confidence
#         filtered_detections = [det for det in fused if det["label"] != "Animal"]
#     else:
#         filtered_detections = fused

#     for det in filtered_detections:
#         x1, y1, x2, y2 = det["box"]
#         conf = det["conf"]
#         label = det["label"]
#         votes = det.get("votes", 1)
#         color = color_map[label]
#         thickness = int(2 + conf * 3)

#         cv2.rectangle(output_img, (x1, y1), (x2, y2), color, thickness)

#         text = f"{label} {conf:.2f}"
#         if votes > 1:
#             text += f" (v:{votes})"

#         (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
#         overlay = output_img.copy()
#         cv2.rectangle(overlay, (x1, y1 - text_h - 12), (x1 + text_w + 10, y1), color, -1)
#         output_img = cv2.addWeighted(output_img, 0.7, overlay, 0.3, 0)

#         cv2.putText(output_img, text, (x1 + 5, y1 - 6),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

#     return output_img



# @app.post("/upload_video/")
# async def upload_file(file: UploadFile = File(...)):
#     session_id = str(uuid.uuid4())
#     upload_path = os.path.join(UPLOAD_DIR, session_id)
#     result_path = os.path.join(RESULT_DIR, session_id)
#     os.makedirs(upload_path, exist_ok=True)
#     os.makedirs(result_path, exist_ok=True)

#     file_path = os.path.join(upload_path, file.filename)
#     with open(file_path, "wb") as buffer:
#         shutil.copyfileobj(file.file, buffer)

#     ext = file.filename.lower().split('.')[-1]

#     if ext in ["mp4", "mov", "avi", "mkv"]:
#         cap = cv2.VideoCapture(file_path)
#         if not cap.isOpened():
#             raise HTTPException(status_code=400, detail="Could not open video file")

#         width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#         fps = cap.get(cv2.CAP_FPS) or 30.0
#         fourcc = cv2.VideoWriter_fourcc(*'mp4v')

#         forest_out_path = os.path.join(result_path, "forest_segmentation.mp4")
#         detect_out_path = os.path.join(result_path, "detections.mp4")

#         forest_out = cv2.VideoWriter(forest_out_path, fourcc, fps, (width, height))
#         detect_out = cv2.VideoWriter(detect_out_path, fourcc, fps, (width, height))

#         frame_count = 0
#         print("⚡ SMART mode optimized for production use!")
#         print("💡 Processing every 5th frame for videos")

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             if frame_count % 5 == 0:
#                 print(f"Processing frame {frame_count}...")
#                 forest_frame = smart_forest_segmentation(frame)
#                 detect_frame = smart_max_detection(frame)
#             else:
#                 forest_frame = frame
#                 detect_frame = frame

#             forest_out.write(forest_frame)
#             detect_out.write(detect_frame)

#             frame_count += 1

#         cap.release()
#         forest_out.release()
#         detect_out.release()

#         outputs = {
#             "forest_segmentation": forest_out_path,
#             "detections": detect_out_path
#         }

#     else:
#         img = cv2.imread(file_path)
#         if img is None:
#             raise HTTPException(status_code=400, detail="Uploaded image is invalid or corrupted")

#         print("🚀 Starting SMART forest segmentation...")
#         forest_img = smart_forest_segmentation(img)

#         print("🚀 Starting SMART MAX detection...")
#         detect_img = smart_max_detection(img)

#         forest_path = os.path.join(result_path, "forest_segmentation.jpg")
#         detect_path = os.path.join(result_path, "detections.jpg")

#         cv2.imwrite(forest_path, forest_img)
#         cv2.imwrite(detect_path, detect_img)

#         outputs = {
#             "forest_segmentation": forest_path,
#             "detections": detect_path
#         }

#     return {
#         "uuid": session_id,
#         "filename": file.filename,
#         "message": "✅ SMART MAXIMUM ACCURACY processing complete! 🔥⚡",
#         "results_folder": result_path,
#         "outputs": outputs
#     }
