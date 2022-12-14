import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)
if not cap.isOpened :
    print("erreur ")
    exit(0)

while True:

#while(cap.isOpened()):
  # Capture frame-by-frame
  ret, frame = cap.read()
  if ret == True:
 
    cv2.imshow('Frame',frame)
    red = frame[:,:,2]
    mask = cv2.inRange(red, 180, 250)
    cv2.imshow('Frame',mask)

    masked_image = np.copy(red)
    masked_image[mask == 0] = 0
    cv2.imshow('Frame',masked_image)

    hist = np.sum(masked_image/255, axis=0)
    print(hist.shape)
    print(np.argmax(hist))
    plt.plot(hist)
    plt.show()

    if cv2.waitKey(1)&0xFF == ord('0') :
     break
 
 
# When everything done, release the video capture object
cap.release() 
 
# Closes all the frames
cv2.destroyAllWindows() 