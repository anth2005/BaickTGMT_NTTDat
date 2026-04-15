import cv2
import numpy as np
import math

# 1. Định nghĩa kích thước ảnh (Yêu cầu: 1024x768)
M = 768   # Chiều cao (y)
N = 1024  # Chiều rộng (x)

# 2. Tạo hình với pixel ngẫu nhiên (Yêu cầu 1)
# np.random.randint(0, 256) sẽ tạo ra số lớn nhất là 255 -> KHÔNG BỊ LỖI uint8
img = np.random.randint(0, 256, (M, N, 3), dtype=np.uint8)

# 3. Vẽ đường chéo đỏ (Yêu cầu 2)
for j in range(N): 
    i = int(j * (M / N)) 
    if i < M:
        img[i, j] = [0, 0, 255] # [Blue, Green, Red]

# 4. Vẽ đồng hồ La Mã (Yêu cầu 3)
center_x, center_y = N // 2, M // 2
radius = 300 
romans = ["III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "I", "II"]

for k in range(12):
    angle = math.radians(k * 30)
    x = int(center_x + radius * math.cos(angle))
    y = int(center_y + radius * math.sin(angle))
    
    # Vẽ chữ màu trắng (255, 255, 255)
    cv2.putText(img, romans[k], (x - 40, y + 15), 
                cv2.FONT_HERSHEY_DUPLEX, 1.5, (255, 255, 255), 2)

cv2.imshow('Fixed Overflow and Clock', img)
cv2.waitKey(0)
cv2.destroyAllWindows()