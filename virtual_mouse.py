
import time

import cv2 as cv
import mediapipe as mp
import numpy as np
import pyautogui

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


# =========================
# 1. CONFIGURATION
# =========================

MODEL_PATH = "hand_landmarker.task"
WINDOW_NAME = "NITHISH | VIRTUAL MOUSE"

screen_width, screen_height = pyautogui.size()

pyautogui.FAILSAFE = True
pyautogui.PAUSE = 0.03

smooth_factor = 0.2
smooth_x = None
smooth_y = None

click_threshold = 0.05
was_pinching = False
last_click_time = 0.0
click_cooldown = 0.4


# =========================
# 2. MEDIAPIPE SETUP
# =========================

base_options = python.BaseOptions(
    model_asset_path=MODEL_PATH
)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7,
    running_mode=vision.RunningMode.VIDEO
)


# =========================
# 3. DESIGN HELPERS
# =========================

def draw_text(frame, text, position, color, scale=0.6):
    cv.putText(
        frame,
        text,
        position,
        cv.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv.LINE_AA
    )


def draw_dashboard(frame, active, pinching):
    h, w = frame.shape[:2]

    # Dark header
    cv.rectangle(frame, (0, 0), (w, 100), (35, 25, 25), cv.FILLED)

    # Project title
    draw_text(
        frame, "VIRTUAL MOUSE",
        (20, 35), (255, 255, 255), 0.85
    )

    draw_text(
        frame, "HAND CONTROL SYSTEM",
        (22, 65), (220, 200, 120), 0.48
    )

    # Status panel
    if active:
        status = "VIRTUAL MOUSE ACTIVE"
        status_color = (70, 230, 90)
        detail = "Hand detected - move your index finger"
    else:
        status = "WAITING FOR HAND"
        status_color = (0, 180, 255)
        detail = "Show your hand to start"

    cv.rectangle(
        frame, (10, 115), (w - 10, 175),
        (35, 35, 35), cv.FILLED
    )

    cv.circle(frame, (28, 140), 7, status_color, cv.FILLED)

    draw_text(frame, status, (45, 145), status_color, 0.55)
    draw_text(frame, detail, (20, 163), (230, 230, 230), 0.38)

    # Gesture status
    if active and pinching:
        draw_text(
            frame, "LEFT CLICK!",
            (20, h - 25), (0, 255, 100), 0.7
        )
    else:
        draw_text(
            frame, "Pinch thumb + index to click",
            (20, h - 25), (255, 255, 255), 0.48
        )

    draw_text(
        frame, "Press Q to quit",
        (w - 155, h - 25), (180, 180, 180), 0.42
    )


def draw_virtual_cursor(frame, x, y):
    # Custom cursor symbol: target + crosshair
    cv.circle(frame, (x, y), 18, (255, 80, 220), 2)
    cv.circle(frame, (x, y), 5, (0, 255, 255), cv.FILLED)

    cv.line(
        frame, (x - 25, y), (x + 25, y),
        (255, 80, 220), 2
    )
    cv.line(
        frame, (x, y - 25), (x, y + 25),
        (255, 80, 220), 2
    )


# =========================
# 4. OPEN WEBCAM
# =========================

cap = cv.VideoCapture(0)

if not cap.isOpened():
    raise RuntimeError("Could not open webcam.")

cv.namedWindow(WINDOW_NAME, cv.WINDOW_NORMAL)
cv.resizeWindow(WINDOW_NAME, 640, 480)

# Place preview in the top-right corner
cv.moveWindow(
    WINDOW_NAME,
    max(0, screen_width - 660),
    40
)

# Ask Windows/OpenCV to keep the preview above other windows.
# This property may not be supported by every OpenCV backend.
try:
    cv.setWindowProperty(
        WINDOW_NAME,
        cv.WND_PROP_TOPMOST,
        1
    )
except cv.error:
    print("Always-on-top is not supported by this OpenCV backend.")


# =========================
# 5. MAIN PROGRAM
# =========================

try:
    with vision.HandLandmarker.create_from_options(
        options
    ) as detector:

        start_time = time.monotonic()

        while True:
            success, frame = cap.read()

            if not success:
                print("Could not read webcam frame.")
                break

            frame = cv.flip(frame, 1)
            h, w = frame.shape[:2]

            rgb_frame = cv.cvtColor(
                frame, cv.COLOR_BGR2RGB
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            timestamp_ms = int(
                (time.monotonic() - start_time) * 1000
            )

            results = detector.detect_for_video(
                mp_image, timestamp_ms
            )

            hand_detected = bool(results.hand_landmarks)
            is_pinching = False

            if hand_detected:
                hand = results.hand_landmarks[0]

                index_tip = hand[8]
                thumb_tip = hand[4]

                # Index fingertip in preview pixels
                x = int(index_tip.x * w)
                y = int(index_tip.y * h)

                # Map finger position to screen
                screen_x = int(
                    np.interp(
                        x, [0, w - 1],
                        [0, screen_width - 1]
                    )
                )

                screen_y = int(
                    np.interp(
                        y, [0, h - 1],
                        [0, screen_height - 1]
                    )
                )

                # Smooth cursor movement
                if smooth_x is None:
                    smooth_x = screen_x
                    smooth_y = screen_y
                else:
                    smooth_x += (
                        screen_x - smooth_x
                    ) * smooth_factor

                    smooth_y += (
                        screen_y - smooth_y
                    ) * smooth_factor

                # Move system mouse
                pyautogui.moveTo(
                    int(smooth_x),
                    int(smooth_y)
                )

                # Draw custom symbol in webcam preview
                draw_virtual_cursor(frame, x, y)

                # Calculate pinch distance
                distance = np.hypot(
                    index_tip.x - thumb_tip.x,
                    index_tip.y - thumb_tip.y
                )

                is_pinching = distance < click_threshold

                # Click once when pinch begins
                now = time.monotonic()

                if (
                    is_pinching
                    and not was_pinching
                    and now - last_click_time >= click_cooldown
                ):
                    pyautogui.click()
                    last_click_time = now
                    print("Left click detected!")

                was_pinching = is_pinching

            else:
                # Reset click state if hand disappears
                was_pinching = False

            # Draw dashboard last so it stays visible
            draw_dashboard(
                frame,
                hand_detected,
                is_pinching
            )

            cv.imshow(WINDOW_NAME, frame)

            # Process keyboard and window events
            key = cv.waitKey(1) & 0xFF

            if key == ord("q"):
                break

finally:
    cap.release()
    cv.destroyAllWindows()
    print("Virtual Mouse stopped safely.")
