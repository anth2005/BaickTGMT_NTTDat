import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# 1. Khởi tạo module Hand Tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 2. Khởi tạo module Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

import time
# Mở webcam
cap = cv2.VideoCapture(0)
time.sleep(2) # Chờ webcam khởi động

print("Đang chạy Webcam... Nhấn phím 'ESC' để thoát.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Không thể đọc từ webcam.")
        break

    # Để tăng tốc độ xử lý, chuyển ảnh sang RGB và đánh dấu chỉ đọc
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Chạy mô hình để nhận diện
    hand_results = hands.process(image)
    face_results = face_mesh.process(image)

    # Đánh dấu ảnh có thể ghi lại để vẽ các landmark và chuyển lại sang BGR
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # ===== Vẽ Hand Tracking =====
    if hand_results.multi_hand_landmarks:
        for hand_landmarks in hand_results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

    # ===== Vẽ Face Mesh =====
    if face_results.multi_face_landmarks:
        for face_landmarks in face_results.multi_face_landmarks:
            # Vẽ lưới khuôn mặt
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_tesselation_style()
            )
            # Vẽ viền khuôn mặt
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            # Vẽ vùng mắt (Iris)
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_iris_connections_style()
            )

    # Hiển thị (Lật ngược ảnh để giống như soi gương)
    cv2.imshow('MediaPipe Demo: Hand Tracking & Face Mesh', cv2.flip(image, 1))
    
    # Nhấn ESC để thoát
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
