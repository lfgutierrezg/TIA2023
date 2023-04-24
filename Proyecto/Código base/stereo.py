from camera import Camera
import depthai as dai
import cv2 as cv
import numpy as np

camera = Camera()
device = camera.createStereoCamera()

# Get output queues. 
leftQueue = device.getOutputQueue(name="left", maxSize=1)
# maxSize -> Frame holding capacity
rightQueue = device.getOutputQueue(name="right", maxSize=1)

# Set display window name
cv.namedWindow("Stereo Pair")
# Variable used to toggle between side by side view and one frame view.
sideBySide = False
images = 0

while True:
    # Get left frame
    leftFrame = camera.getFrame(leftQueue)
    # Get right frame 
    rightFrame = camera.getFrame(rightQueue)
        
    if sideBySide:
        # Show side by side view
        imOut = np.hstack((leftFrame, rightFrame))
    else : 
        # Show overlapping frames
        imOut = np.uint8(leftFrame/2 + rightFrame/2)
        
    # Display output image
    cv.imshow("Stereo Pair", imOut)
    cv.imwrite(f"./images/{images}.jpg", imOut)
    images+=1
    # Check for keyboard input
    key = cv.waitKey(1)
    if key == ord('q'):
        # Quit when q is pressed
        break
    elif key == ord('t'):
        # Toggle display when t is pressed
        sideBySide = not sideBySide