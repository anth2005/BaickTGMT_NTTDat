import cv2
from ultralytics import YOLO
import time
import os
import argparse  # Thêm thư viện này

# ===== MODEL =====
model = YOLO("yolov8n.pt")

# ===== CONFIG =====
ROI = (0, 0, 1280, 720)
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

def point_in_roi(cx, cy, roi):
    x1, y1, x2, y2 = roi
    return x1 <= cx <= x2 and y1 <= cy <= y2

def main(video_path):
    # Kiểm tra file có tồn tại không trước khi chạy
    if not os.path.exists(video_path):
        print(f"Error: Khong tim thay file video tai {video_path}")
        return

    cap = cv2.VideoCapture(video_path)
    delay = 1 

    memory = {}          
    counted_ids = set()  
    count = {"car": 0, "motorcycle": 0, "bus": 0, "truck": 0}

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1280, 720))
        results = model.track(
            frame, persist=True, tracker="bytetrack.yaml",
            conf=0.5, iou=0.6, verbose=False
        )

        if results[0].boxes is None or results[0].boxes.id is None:
            cv2.imshow("Vehicle Counter Stable", frame)
            if cv2.waitKey(delay) & 0xFF == ord("q"): break
            continue

        boxes = results[0].boxes.xyxy.cpu().numpy()
        clss = results[0].boxes.cls.int().cpu().numpy()
        ids = results[0].boxes.id.int().cpu().numpy()
        confs = results[0].boxes.conf.cpu().numpy()

        for box, cls_id, obj_id, conf in zip(boxes, clss, ids, confs):
            if cls_id not in VEHICLE_CLASSES or conf < 0.4:
                continue

            x1, y1, x2, y2 = map(int, box)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            if not point_in_roi(cx, cy, ROI):
                continue

            # Loc theo kich thuoc de tranh nham lan
            if (x2 - x1) * (y2 - y1) < 1800:
                continue

            label = VEHICLE_CLASSES[cls_id]
            obj_id = int(obj_id)

            if obj_id not in memory:
                memory[obj_id] = []
            memory[obj_id].append(label)
            if len(memory[obj_id]) > 7:
                memory[obj_id].pop(0)

            final_class = max(set(memory[obj_id]), key=memory[obj_id].count)

            if obj_id not in counted_ids:
                count[final_class] += 1
                counted_ids.add(obj_id)

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{final_class} ID:{obj_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # UI Overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (20, 20), (320, 180), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        cv2.putText(frame, "VEHICLE COUNT", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        y = 90
        for k, v in count.items():
            cv2.putText(frame, f"{k}: {v}", (40, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)
            y += 25

        cv2.imshow("Vehicle Counter Stable", frame)
        if cv2.waitKey(delay) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Khởi tạo bộ phân tích tham số dòng lệnh
    parser = argparse.ArgumentParser(description="Vehicle Counting with YOLOv8")
    
    # Cho phép truyền thẳng tên video mà không cần thêm --input hay -i
    parser.add_argument("video", nargs="?", type=str, default=None, help="Đường dẫn đến file video")
    
    args = parser.parse_args()

    # Nếu người dùng nhập tham số, dùng tham số đó, nếu không dùng mặc định
    if args.video:
        video_path = args.video
    else:
        # File mặc định như cũ của bạn
        video_path = os.path.join(os.path.dirname(__file__), "..", "haha.mp4")

    main(video_path)