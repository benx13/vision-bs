import cv2
import numpy as np

lo  = np.array([80, 255, 255])
hi  = np.array([120, 255, 255])

def detect_inrange(image, surface): 
    p = []
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = cv2.blur(image, (5, 5)) 
    mask = cv2.inRange(image, lo, hi)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)
    elems = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    elems = sorted(elems, key=lambda x:cv2.contourArea(x), reverse=True) 
    for elem in elems:
        if cv2.contourArea(elem) > surface:
            ((x, y), _) = cv2.minEnclosingCircle(elem)
            p.append((int(x), int(y)))
        else:
            break 
    return image, mask, p

videoCap = cv2.VideoCapture(0)
while(True):
    ret, frame = videoCap.read()
    cv2.flip(frame, 1, frame)
    image, mask, p = detect_inrange(frame, 200) 
    for i in p:
        cv2.circle(image, i, 10, (0, 255, 0), 10)
        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (255, 0, 0)
        thickness = 2
        image = cv2.putText(image, str(i), i, font, 
                    fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow('image', image, )

    if mask is not None:
        cv2.imshow('mask', mask)
    if cv2.waitKey(10)&0xFF == ord('0') :
        break

videoCap.release()
cv2.destroyAllWindows()