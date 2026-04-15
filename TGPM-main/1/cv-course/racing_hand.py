"""
Racing Hand – Đua xe điều khiển bằng bàn tay
Di chuyển bàn tay trái / phải để lái xe tránh chướng ngại vật.
"""
import pygame
import random
import time
import threading
import cv2
import mediapipe as mp
import math
from pygame.locals import *

# ========================
# HẰNG SỐ
# ========================
SW, SH = 480, 680          # kích thước màn hình game
FPS    = 60
ROAD_LEFT  = 80            # mép trái đường
ROAD_RIGHT = 400           # mép phải đường
LANE_W = (ROAD_RIGHT - ROAD_LEFT) / 3  # 3 làn

# ========================
# MEDIAPIPE – HAND TRACKING (thread riêng)
# ========================
hand_x_norm = 0.5   # vị trí tay ngang (0.0 = trái, 1.0 = phải, 0.5 = giữa)
hand_y_norm = 0.5   # vị trí tay dọc (0.0 = trên, 1.0 = dưới)
hand_visible = False

def camera_thread():
    global hand_x_norm, hand_visible
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
        max_num_hands=1
    )
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    time.sleep(1.5)

    # Trail cho cổ tay
    trail = []
    TRAIL_LEN = 12

    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)   # mirror
        h, w = frame.shape[:2]

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = hands.process(rgb)
        rgb.flags.writeable = True
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if result.multi_hand_landmarks:
            for hl in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

                # Lấy tọa độ X trung bình của cổ tay + gốc các ngón
                pts = [0, 5, 9, 13, 17]  # wrist + ngón
                avg_x = sum(hl.landmark[i].x for i in pts) / len(pts)
                avg_y = sum(hl.landmark[i].y for i in pts) / len(pts)
                hand_x_norm  = avg_x        # 0 = trái, 1 = phải (đã flip)
                hand_y_norm  = avg_y        # 0 = trên, 1 = dưới
                hand_visible = True

                # Trail
                cx = int(avg_x * w)
                cy = int(hl.landmark[0].y * h)
                trail.append((cx, cy))
                if len(trail) > TRAIL_LEN:
                    trail.pop(0)

                # Vẽ trail
                for i in range(1, len(trail)):
                    alpha = int(255 * i / TRAIL_LEN)
                    cv2.line(frame, trail[i-1], trail[i], (0, alpha, 255 - alpha), 3)

                # Hướng dẫn lái
                col = (0, 200, 255)
                label = f"X: {avg_x:.2f}  {'<<< TRAI' if avg_x < 0.35 else ('PHAI >>>' if avg_x > 0.65 else 'GIUA')}"
                cv2.putText(frame, label, (15, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, col, 2)
        else:
            hand_visible = False
            trail.clear()
            cv2.putText(frame, "Di chuyen ban tay vao khung hinh", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (100, 100, 100), 2)

        # Thanh chỉ vị trí tay
        bar_y = h - 30
        cv2.rectangle(frame, (0, bar_y - 10), (w, bar_y + 10), (40, 40, 40), -1)
        pos_x = int(hand_x_norm * w)
        cv2.circle(frame, (pos_x, bar_y), 12, (0, 220, 255), -1)

        cv2.imshow("Cam – Di chuyen tay de lai xe (ESC thoat)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()

# ========================
# VẼ CÁC ĐỐI TƯỢNG GAME
# ========================
def draw_car(surf, x, y, color, w=48, h=85):
    """Vẽ xe nhìn từ trên xuống."""
    # Thân xe
    body = pygame.Rect(x - w//2, y - h//2, w, h)
    pygame.draw.rect(surf, color, body, border_radius=8)
    # Kính trước
    pygame.draw.rect(surf, (180, 230, 255),
                     (x - w//2 + 4, y - h//2 + 8, w - 8, 14), border_radius=4)
    # Kính sau
    pygame.draw.rect(surf, (180, 230, 255),
                     (x - w//2 + 4, y + h//2 - 22, w - 8, 10), border_radius=4)
    # Bánh xe
    wheel_color = (30, 30, 30)
    pygame.draw.rect(surf, wheel_color, (x - w//2 - 5, y - h//2 + 6, 9, 16), border_radius=3)
    pygame.draw.rect(surf, wheel_color, (x + w//2 - 4, y - h//2 + 6, 9, 16), border_radius=3)
    pygame.draw.rect(surf, wheel_color, (x - w//2 - 5, y + h//2 - 22, 9, 16), border_radius=3)
    pygame.draw.rect(surf, wheel_color, (x + w//2 - 4, y + h//2 - 22, 9, 16), border_radius=3)

def draw_road(surf, road_scroll):
    """Vẽ mặt đường và vạch kẻ."""
    # Lề
    surf.fill((60, 80, 50))
    # Đường nhựa
    pygame.draw.rect(surf, (55, 55, 60), (ROAD_LEFT, 0, ROAD_RIGHT - ROAD_LEFT, SH))
    # Vạch lề trắng
    pygame.draw.rect(surf, (240, 240, 240), (ROAD_LEFT, 0, 5, SH))
    pygame.draw.rect(surf, (240, 240, 240), (ROAD_RIGHT - 5, 0, 5, SH))

    # Vạch giữa làn (nét đứt)
    dash_h, gap = 40, 30
    lane_xs = [ROAD_LEFT + LANE_W, ROAD_LEFT + 2 * LANE_W]
    for lx in lane_xs:
        y_start = int(-road_scroll % (dash_h + gap))
        while y_start < SH:
            pygame.draw.rect(surf, (220, 200, 60),
                             (int(lx) - 2, y_start, 4, dash_h))
            y_start += dash_h + gap

    # Cây bên lề (đơn giản)
    tree_xs = [20, 45, ROAD_RIGHT + 15, ROAD_RIGHT + 45]
    for tx in tree_xs:
        y_start = int(-road_scroll * 0.8 % 160)
        while y_start < SH:
            pygame.draw.circle(surf, (34, 120, 34), (tx, y_start + 30), 18)
            pygame.draw.rect(surf, (100, 70, 40), (tx - 3, y_start + 30, 6, 24))
            y_start += 160

def draw_obstacle(surf, obs):
    draw_car(surf, obs['x'], obs['y'], obs['color'])

# ========================
# VÒNG LẶP CHÍNH
# ========================
def main():
    # Bật camera thread
    t = threading.Thread(target=camera_thread, daemon=True)
    t.start()

    pygame.init()
    pygame.mixer.init(44100, -16, 2, 512)
    screen = pygame.display.set_mode((SW, SH))
    pygame.display.set_caption("Racing Hand – Lai xe bang ban tay")
    clock  = pygame.time.Clock()

    font_big   = pygame.font.SysFont("Arial", 52, bold=True)
    font_mid   = pygame.font.SysFont("Arial", 32, bold=True)
    font_small = pygame.font.SysFont("Arial", 24)

    # Màu xe của người chơi
    PLAYER_COLOR  = (255, 60, 60)
    OBS_COLORS = [(50, 150, 255), (255, 200, 0), (80, 220, 80),
                  (200, 80, 220), (255, 140, 0)]

    # ---- Màn hình chờ ----
    waiting = True
    anim_y  = 0.0
    while waiting:
        clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit(); return
            if event.type == KEYDOWN and event.key == K_SPACE:
                waiting = False

        # Kích hoạt bằng tay (di chuyển sang trái hoặc phải mạnh)
        if hand_visible and (hand_x_norm < 0.3 or hand_x_norm > 0.7):
            waiting = False

        anim_y = (anim_y + 0.05) % (2 * math.pi)

        screen.fill((20, 20, 30))
        draw_road(screen, pygame.time.get_ticks() * 0.05)

        # Title
        title = font_big.render("RACING HAND", True, (255, 220, 60))
        screen.blit(title, title.get_rect(center=(SW//2, 160)))

        sub = font_small.render("Di tay TRAI/PHAI de bat dau", True, (200, 200, 200))
        screen.blit(sub, sub.get_rect(center=(SW//2, 215)))

        # Xe demo nhảy lên xuống
        demo_y = SH // 2 + int(math.sin(anim_y) * 15)
        draw_car(screen, SW//2, demo_y, PLAYER_COLOR)

        # Thanh vị trí tay
        bar_rect = pygame.Rect(ROAD_LEFT, SH - 50, ROAD_RIGHT - ROAD_LEFT, 16)
        pygame.draw.rect(screen, (80, 80, 80), bar_rect, border_radius=8)
        if hand_visible:
            px = ROAD_LEFT + int(hand_x_norm * (ROAD_RIGHT - ROAD_LEFT))
            pygame.draw.circle(screen, (0, 220, 255), (px, SH - 42), 12)
            label = font_small.render("TAY PHAT HIEN!", True, (0, 220, 255))
        else:
            label = font_small.render("Dua tay vao camera...", True, (160, 160, 160))
        screen.blit(label, label.get_rect(center=(SW//2, SH - 20)))

        pygame.display.update()

    # ---- Trạng thái game ----
    road_scroll = 0.0
    speed       = 4.0    # tốc độ ban đầu (pixel/frame)
    score       = 0
    player_x    = SW // 2
    player_y    = SH - 120

    obstacles = []
    spawn_timer = 0
    spawn_interval = 90   # frame giữa 2 lần spawn

    running = True
    while running:
        dt = clock.tick(FPS)
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit(); return
            if event.type == KEYDOWN and event.key == K_ESCAPE:
                running = False

        # ---- Điều khiển xe ----
        if hand_visible:
            # Map hand_x_norm (0..1) -> xe
            target_x = ROAD_LEFT + 25 + hand_x_norm * (ROAD_RIGHT - ROAD_LEFT - 50)
            # Map hand_y_norm (0..1) -> xe (y: 100 .. SH-50)
            target_y = 100 + hand_y_norm * (SH - 150)
        else:
            # Fallback bàn phím
            keys = pygame.key.get_pressed()
            if keys[K_LEFT]:  player_x -= 5
            if keys[K_RIGHT]: player_x += 5
            if keys[K_UP]:    player_y -= 5
            if keys[K_DOWN]:  player_y += 5
            target_x = player_x
            target_y = player_y

        # Smooth follow tay
        player_x += (target_x - player_x) * 0.18
        player_y += (target_y - player_y) * 0.18
        
        player_x = max(ROAD_LEFT + 25, min(ROAD_RIGHT - 25, player_x))
        player_y = max(80, min(SH - 60, player_y))

        # ---- Tốc độ tăng dần ----
        speed = 4.0 + score * 0.015
        road_scroll += speed

        # ---- Sinh chướng ngại vật ----
        spawn_timer += 1
        if spawn_timer >= spawn_interval:
            spawn_timer = 0
            spawn_interval = max(40, spawn_interval - 1)
            lane = random.randint(0, 2)
            obs_x = ROAD_LEFT + LANE_W * lane + LANE_W / 2
            obstacles.append({
                'x': obs_x,
                'y': -40,
                'color': random.choice(OBS_COLORS),
                'speed': speed * random.uniform(0.7, 1.1)
            })

        # ---- Cập nhật obstacle ----
        for obs in obstacles:
            obs['y'] += obs['speed']
        obstacles = [o for o in obstacles if o['y'] < SH + 60]

        score += 1

        # ---- Vẽ ----
        draw_road(screen, road_scroll)

        for obs in obstacles:
            draw_obstacle(screen, obs)

        draw_car(screen, int(player_x), player_y, PLAYER_COLOR)

        # HUD điểm + tốc độ
        sc_surf = font_mid.render(f"Score: {score // 10}", True, (255, 255, 255))
        sp_surf = font_small.render(f"Speed: {int(speed * 10)} km/h", True, (200, 200, 200))
        screen.blit(sc_surf, (10, 10))
        screen.blit(sp_surf, (10, 48))

        # Thanh vị trí tay (HUD bên phải)
        bar_w = 20
        bar_h = 120
        bx    = SW - 35
        by    = SH // 2 - bar_h // 2
        pygame.draw.rect(screen, (60, 60, 60), (bx, by, bar_w, bar_h), border_radius=6)
        dot_y = by + int(hand_x_norm * bar_h)
        if hand_visible:
            pygame.draw.circle(screen, (0, 220, 255), (bx + bar_w//2, dot_y), 9)
        else:
            hw_txt = font_small.render("?", True, (160, 160, 160))
            screen.blit(hw_txt, (bx + 4, by + bar_h//2 - 10))

        pygame.display.update()

        # ---- Va chạm (Sử dụng Rect để chính xác hơn) ----
        pw, ph = 48, 85
        player_rect = pygame.Rect(int(player_x) - pw//2, player_y - ph//2, pw, ph)
        for obs in obstacles:
            obs_rect = pygame.Rect(int(obs['x']) - pw//2, int(obs['y']) - ph//2, pw, ph)
            if player_rect.colliderect(obs_rect):
                running = False
                break

    # ---- Game Over ----
    pygame.mixer.Sound  # (không có hit.wav trong racing – dùng beep đơn giản)

    final_score = score // 10
    show_go = True
    go_timer = 0
    prev_visible = hand_visible

    while show_go:
        clock.tick(FPS)
        go_timer += 1
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit(); return
            if event.type == KEYDOWN and event.key in (K_SPACE, K_r):
                show_go = False

        # Nắm tay hoặc di tay sang 1 bên để chơi lại
        if hand_visible and not prev_visible:
            show_go = False
        prev_visible = hand_visible

        # Overlay mờ
        overlay = pygame.Surface((SW, SH), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 160))
        screen.blit(overlay, (0, 0))

        go_surf  = font_big.render("GAME OVER", True, (255, 60, 60))
        sc_surf  = font_mid.render(f"Score: {final_score}", True, (255, 220, 60))
        rst_surf = font_small.render("Space / R hoac di chuyen tay de choi lai", True, (200, 200, 200))
        screen.blit(go_surf,  go_surf.get_rect(center=(SW//2, SH//2 - 60)))
        screen.blit(sc_surf,  sc_surf.get_rect(center=(SW//2, SH//2)))
        screen.blit(rst_surf, rst_surf.get_rect(center=(SW//2, SH//2 + 56)))
        pygame.display.update()

    main()   # Chơi lại

if __name__ == "__main__":
    main()
