import cv2 as cv
import numpy as np
import math

# Đường dẫn video (đảm bảo tồn tại)
video_path = r"C:\Users\Anth1\Downloads\1\1\bang_chuyen.mp4"
cap = cv.VideoCapture(video_path)
if not cap.isOpened():
    print("Không mở được video: ", video_path)
    exit()

# Tìm vạch đỏ (giả sử có trong khung đầu)
ret, first_frame = cap.read()
if not ret:
    print("Không đọc được khung đầu của video")
    exit()

hsv = cv.cvtColor(first_frame, cv.COLOR_BGR2HSV)
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])
mask = cv.inRange(hsv, lower_red1, upper_red1) + cv.inRange(hsv, lower_red2, upper_red2)
contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
line_x = None
for cnt in contours:
    if cv.contourArea(cnt) > 500:
        x, y, w, h = cv.boundingRect(cnt)
        line_x = x + w // 2
        break
if line_x is None:
    print("Không tìm thấy vạch đỏ trong video")
    exit()
print("Vạch đỏ tại X =", line_x)
# Đặt lại vị trí đọc video sau khung đầu
cap.set(cv.CAP_PROP_POS_FRAMES, 1)

# Theo dõi đối tượng
next_id = 0
objects = {}  # id -> (x, y, unseen_frames)
counted_ids = set()
count = 0
MAX_UNSEEN = 5

def distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

while True:
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.medianBlur(gray, 5)

    # Phát hiện vòng tròn
    circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, dp=1, minDist=20,
                             param1=50, param2=30, minRadius=5, maxRadius=80)
    detections = []  # list of (x, y, r)
    if circles is not None:
        circles = np.uint16(np.round(circles))
        for c in circles[0, :]:
            detections.append((int(c[0]), int(c[1]), int(c[2])))

    # Cập nhật đối tượng hiện có
    new_objects = {}
    used_det = set()
    for obj_id, (px, py, unseen) in objects.items():
        best_idx = None
        best_dist = 9999
        for i, (cx, cy, cr) in enumerate(detections):
            if i in used_det:
                continue
            d = distance((cx, cy), (px, py))
            if d < 40 and d < best_dist:
                best_dist = d
                best_idx = i
        if best_idx is not None:
            cx, cy, cr = detections[best_idx]
            new_objects[obj_id] = (cx, cy, 0)
            used_det.add(best_idx)
            # Kiểm tra qua vạch
            if obj_id not in counted_ids and px <= line_x < cx:
                count += 1
                counted_ids.add(obj_id)
        else:
            # Không thấy, tăng bộ đếm mất dấu
            if unseen < MAX_UNSEEN:
                new_objects[obj_id] = (px, py, unseen + 1)

    # Đăng ký các detections mới chưa gán
    for i, (cx, cy, cr) in enumerate(detections):
        if i not in used_det:
            new_objects[next_id] = (cx, cy, 0)
            next_id += 1

    objects = new_objects

    # Vẽ
    cv.line(frame, (line_x, 0), (line_x, frame.shape[0]), (0, 255, 255), 2)
    for obj_id, (cx, cy, unseen) in objects.items():
        if unseen == 0:
            cv.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            cv.putText(frame, f"ID {obj_id}", (cx - 10, cy - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv.putText(frame, f"Count: {count}", (20, 50), cv.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 2)
    cv.imshow("Tracking", frame)
    if cv.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
print("Tổng số vòng tròn qua vạch:", count)