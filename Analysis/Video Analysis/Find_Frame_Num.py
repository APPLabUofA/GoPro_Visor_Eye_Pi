import numpy as np
import cv2
import pandas as pd

cap = cv2.VideoCapture(in_file)
video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
## The folowing script is a more refined version
    # for par 11 = 253956
frame_count = 0
while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
print(str(video_length) + " " + str(frame_count))
# When everything done, release the capture
width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
cap.release()
cv2.destroyAllWindows()