
import matplotlib.pyplot as plt
import numpy as np
import math
import cv2 

#Black and white colormap

#mang pixel values to 0-255 range (8 bits)
M = 255 #1024x768
N = 255
K = 100
# ảnh có MxN pixels 
img = np.zeros((M,N), dtype=np.uint8)
for i in range(M):
    img[i,0:100] = i
# plt.imshow(img, cmap='gray')
# plt.show()
cv2.imshow('Gray Map', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
#Color colormap
img_color = np.zeros((M,N,3), dtype=np.uint8) #Red, Green, Blue

for i in range(M):
    img_color[i,0:64,0] = i  #Red channel
    img_color[i,100:150,1] = i  #Green channel
    img_color[i,200:250,2] = i  #Blue channel
# plt.imshow(img_color)
# plt.show()
# print(img_color)
cv2.imshow('Color Map', img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Random
img_color = np.random.randint(0,256,(M,N,3), dtype=np.uint8)
# plt.imshow(img_color)
# plt.show()
cv2.imshow('Random Color Map', img_color)
cv2.waitKey(0)
cv2.destroyAllWindows()

#cross
# Duyệt theo chiều rộng của ảnh
img = np.zeros((M, N, 3), dtype=np.uint8)
for j in range(N): 
    # Tính toán chỉ số hàng i tương ứng để tạo đường chéo
    i = int(j * (M / N)) 
    if i < M:
        img[i, j] = [0, 0, 255]; # Gán pixel tại (hàng i, cột j) thành màu đỏ
# plt.imshow(img)
# plt.show()
cv2.imshow('Cross Line', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

#Roman Numbersl
img = np.zeros((M, N, 3), dtype=np.uint8)
center_x = N // 2
center_y = M // 2
R = 300 # Bán kính phù hợp
romans = ["III", "IV", "V", "VI", "VII", "VIII", "IX", "X", "XI", "XII", "I", "II"]

for k in range(12):
    # Tính góc cho từng số (mỗi số cách nhau 30 độ)
    angle_deg = k * 30 
    angle_rad = math.radians(angle_deg)
    
    # Tính tọa độ x, y
    x = int(center_x + R * math.cos(angle_rad))
    y = int(center_y + R * math.sin(angle_rad))
    
    # In ra terminal để kiểm tra tọa độ nếu vẫn không thấy hình
    # print(f"Số {romans[k]} tại: x={x}, y={y}")
    
    # Vẽ chữ (Dùng màu trắng [255, 255, 255])
    cv2.putText(img, romans[k], (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
# plt.imshow(img)
# plt.show()
cv2.imshow('Roman Numbers', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
# Tạo một bức hình có kích thước 1024x768 với các giá trị pixel ngẫu nhiên
# Vẽ một đường chéo, tô màu đỏ cho đường chéo đó và hiển thị bức hình
# Vẽ lần lượt các chữ số La Mã I đến XII tương tự như trên một chiếc đồng hồ
