import cv2
from ultralytics import YOLO
import time
import math
import os

# ===== LOAD MODEL =====
model = YOLO("yolov8n.pt")

# ===== CONFIG =====
ROI = (200, 200, 1080, 600)
MERGE_DISTANCE = 80
MEMORY_TIME = 5

# COCO classes
VEHICLE_CLASSES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck"
}

# ===== HELPERS =====
def point_in_roi(cx, cy, roi):
    x1, y1, x2, y2 = roi
    return x1 <= cx <= x2 and y1 <= cy <= y2

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

# ===== MAIN =====
def main(video_path):
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    delay = int(1000 / fps) if fps > 0 else 30

    # (track_id -> class history)
    memory = {}  

    # counted IDs
    counted_ids = set()

    # final count
    count = {
        "car": 0,
        "motorcycle": 0,
        "bus": 0,
        "truck": 0
    }

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1280, 720))
        now = time.time()

        # ===== TRACKING =====
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            conf=0.4,
            iou=0.5,
            verbose=False
        )

        cv2.rectangle(frame, (ROI[0], ROI[1]), (ROI[2], ROI[3]), (255, 0, 0), 2)

        if results[0].boxes is not None and results[0].boxes.id is not None:

            boxes = results[0].boxes.xyxy.cpu().numpy()
            clss = results[0].boxes.cls.int().cpu().numpy()
            ids = results[0].boxes.id.int().cpu().numpy()

            for i, (box, cls_id, obj_id) in enumerate(zip(boxes, clss, ids)):

                if cls_id not in VEHICLE_CLASSES:
                    continue

                x1, y1, x2, y2 = map(int, box)
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                if not point_in_roi(cx, cy, ROI):
                    continue

                area = (x2 - x1) * (y2 - y1)
                if area < 1500:
                    continue

                label = VEHICLE_CLASSES[cls_id]
                obj_id = int(obj_id)

                # ===== CLASS STABILITY (MAJORITY VOTE) =====
                if obj_id not in memory:
                    memory[obj_id] = []

                memory[obj_id].append(label)

                if len(memory[obj_id]) > 5:
                    memory[obj_id].pop(0)

                final_class = max(set(memory[obj_id]), key=memory[obj_id].count)

                # ===== COUNT (ONLY ONCE PER ID) =====
                if obj_id not in counted_ids:
                    count[final_class] += 1
                    counted_ids.add(obj_id)

                # ===== DRAW =====
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{final_class} ID:{obj_id}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

        # ===== UI =====
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (320, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(frame, "VEHICLE COUNT", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        y = 90
        for k, v in count.items():
            cv2.putText(frame, f"{k}: {v}", (40, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (200, 255, 200), 2)
            y += 25

        cv2.imshow("FINAL FIXED VEHICLE COUNTER", frame)

        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


# ===== RUN =====
if __name__ == "__main__":
    video_path = os.path.join(os.path.dirname(__file__), "..", "haha.mp4")
    main(video_path)
