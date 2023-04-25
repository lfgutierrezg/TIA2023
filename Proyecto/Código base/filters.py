from camera import Camera
import depthai as dai
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

def getSobel(img):
    
    #Sobel: cv2.Sobel(src, ddepth, dx, dy, ksize)
    sobelx = (cv.Sobel(img,-1,1,0,ksize=3))
    sobely = (cv.Sobel(img,-1,0,1,ksize=3))
    return sobelx+sobely

camera = Camera()
device = camera.createColorCamera(previewSize=(600,600))

# Get output queue
q_rgb = device.getOutputQueue(name="rgb", maxSize=1)

# Set display window name
cv.namedWindow("Color Camera")

images = 0
sobel = False
while True:
    imOut = None
    # Get rgb frame
    in_rgb = q_rgb.tryGet()
    
    if in_rgb is not None:
        
        imOut = camera.getFrame(q_rgb)
        
    # Display output image
    if imOut is not None:
        img_grayscale = cv.cvtColor(imOut, cv.COLOR_RGB2GRAY)

        if sobel:
            filtered_img = getSobel(img_grayscale)
        else: 
            ret, filtered_img = cv.threshold(img_grayscale, 128, 255, cv.THRESH_BINARY)

        cv.imshow("Color Camera", filtered_img)
        cv.imwrite(f"./images/{images}.jpg", filtered_img)
        images+=1
    # Check for keyboard input
    key = cv.waitKey(1)
    if key == ord('q'):
        # Quit when q is pressed
        break
    elif key == ord('t'):
        sobel = not sobel