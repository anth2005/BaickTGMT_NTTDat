import cv2
import mediapipe as mp
import time
import pyautogui

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Hàm tính toán khoảng cách để xác định tay nắm hay mở
def is_hand_closed(hand_landmarks):
    # Lấy tọa độ các điểm mốc
    # 0: cổ tay, 5: gốc ngón trỏ, 8: đầu ngón trỏ, 9: gốc ngón giữa, 12: đầu ngón giữa
    # 13: gốc ngón áp út, 16: đầu ngón áp út, 17: gốc ngón út, 20: đầu ngón út
    points = []
    for lm in hand_landmarks.landmark:
        points.append((lm.x, lm.y))
        
    # Tính khoảng cách y từ đầu ngón đến gốc ngón (trục y hướng xuống dưới)
    # Nếu y của đầu ngón tay lớn hơn (nằm dưới) khối khớp gốc ngón tay -> ngón tay đang gập
    
    fingers_closed = 0
    # Ngón trỏ
    if points[8][1] > points[6][1]: fingers_closed += 1
    # Ngón giữa
    if points[12][1] > points[10][1]: fingers_closed += 1
    # Ngón áp út
    if points[16][1] > points[14][1]: fingers_closed += 1
    # Ngón út
    if points[20][1] > points[18][1]: fingers_closed += 1

    # Trả về True nếu ít nhất 3 ngón tay gập vào (đang nắm tay)
    return fingers_closed >= 3

def main():
    # Khởi tạo module Hand Tracking
    hands = mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )

    cap = cv2.VideoCapture(0)
    time.sleep(2) # Chờ webcam khởi động

    print("Đang chạy Webcam...")
    print("Mở trò chơi Flappy Bird của bạn lên, và nắm tay lại để nhảy!")
    print("Nhấn phím 'ESC' ở cửa sổ camera để thoát.")

    # Biến để tránh spam phím cách liên tục khi bạn vẫn đang giữ tay nắm
    was_closed = False

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            break

        # Để tăng tốc độ xử lý, chuyển ảnh sang RGB
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Chạy mô hình để nhận diện
        hand_results = hands.process(image)

        # Đánh dấu ảnh có thể ghi lại để vẽ các landmark và chuyển lại sang BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # ===== Vẽ Hand Tracking & Kiểm tra Cử chỉ =====
        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                # Vẽ lên màn hình đồ họa
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style()
                )

                # KIỂM TRA TAY NẮM HAY MỞ
                currently_closed = is_hand_closed(hand_landmarks)

                if currently_closed and not was_closed:
                    # Chuyển trạng thái sang nắm TRONG MỘT NHỊP và mô phỏng nhấn phím Space
                    print("Nhảy! (Nhấn phím Space)")
                    pyautogui.press('space')
                    
                    # Hiển thị chữ lên màn hình
                    cv2.putText(image, "JUMP!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                    
                    was_closed = True
                
                elif not currently_closed:
                    # Nếu tay mở ra, reset lại trạng thái chờ nhịp nắm tiếp theo
                    cv2.putText(image, "Dua tay vao...", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                    was_closed = False

        # Hiển thị (Lật ngược ảnh để giống như soi gương)
        cv2.imshow('Flappy Bird Controller', cv2.flip(image, 1))
        
        # Nhấn ESC để thoát
        if cv2.waitKey(5) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
