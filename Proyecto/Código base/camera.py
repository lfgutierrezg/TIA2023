import depthai as dai
import cv2 as cv
import numpy as np

def getFrame(queue):
    frame = queue.get()
    return frame.getCvFrame()

def getMonoCamera(pipeline, isLeft):
    mono = pipeline.createMonoCamera()
    mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    if isLeft:
        mono.setBoardSocket(dai.CameraBoardSocket.LEFT)
    else:
        mono.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    return mono

if __name__ == '__main__':
    pipeline = dai.Pipeline()
    monoR = getMonoCamera(pipeline, isLeft = False)
    monoL = getMonoCamera(pipeline, isLeft = True)
    
    #Set output Xlink for left Camera
    xoutL = pipeline.createXLinkOut()
    xoutL.setStreamName("left")
    
    #Set output Xlink for right Camera
    xoutR = pipeline.createXLinkOut()
    xoutR.setStreamName("right")
    
    # Attach cameras to output Xlink
    monoL.out.link(xoutL.input)
    monoR.out.link(xoutR.input)
    
    with dai.Device(pipeline, usb2Mode=True) as device:
    # Get output queues. 
        leftQueue = device.getOutputQueue(name="left", maxSize=1)
        #maxSize -> Frame holding capacity
        rightQueue = device.getOutputQueue(name="right", maxSize=1)
    
        # Set display window name
        cv.namedWindow("Stereo Pair")
        # Variable used to toggle between side by side view and one frame view. 
        sideBySide = True

        while True:
            # Get left frame
            leftFrame = getFrame(leftQueue)
            # Get right frame 
            rightFrame = getFrame(rightQueue)
        
            if sideBySide:
                # Show side by side view
                imOut = np.hstack((leftFrame, rightFrame))
            else : 
                # Show overlapping frames
                imOut = np.uint8(leftFrame/2 + rightFrame/2)
        
            # Display output image
            cv.imshow("Stereo Pair", imOut)
            
            # Check for keyboard input
            key = cv.waitKey(1)
            if key == ord('q'):
                # Quit when q is pressed
                break
            elif key == ord('t'):
                # Toggle display when t is pressed
                sideBySide = not sideBySide