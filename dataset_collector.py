import cv2
import os
import numpy as np

SAVE_PATH = "data/telugu/ka"   # change label
os.makedirs(SAVE_PATH, exist_ok=True)

cap = cv2.VideoCapture(0)
canvas = np.zeros((480, 640), dtype=np.uint8)
count = 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    cv2.imshow("Draw Character - Press S to Save", canvas)

    key = cv2.waitKey(1)

    if key == ord('d'):  # draw
        x, y = 320, 240
        cv2.circle(canvas, (x, y), 8, 255, -1)

    elif key == ord('s'):  # save
        img = cv2.resize(canvas, (28, 28))
        cv2.imwrite(f"{SAVE_PATH}/{count}.png", img)
        canvas[:] = 0
        count += 1
        print("Saved:", count)

    elif key == 27:
        break

cap.release()
cv2.destroyAllWindows()
