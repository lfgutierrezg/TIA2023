import depthai as dai
import cv2 as cv
import numpy as np


class Camera():

    def __init__(self, usb2mode = True):
        self.usb2mode = usb2mode 

    def createStereoCamera(self):
        pipeline = dai.Pipeline()
        monoR = self.getMonoCamera(pipeline, isLeft = False)
        monoL = self.getMonoCamera(pipeline, isLeft = True)
        
        #Set output Xlink for left Camera
        xoutL = pipeline.createXLinkOut()
        xoutL.setStreamName("left")
            
        #Set output Xlink for right Camera
        xoutR = pipeline.createXLinkOut()
        xoutR.setStreamName("right")
            
        # Attach cameras to output Xlink
        monoL.out.link(xoutL.input)
        monoR.out.link(xoutR.input)

        return dai.Device(pipeline, usb2Mode=self.usb2mode)
    
    def createColorCamera():
        pass

    def getFrame(self, queue):
        frame = queue.get()
        return frame.getCvFrame()

    def getMonoCamera(self, pipeline, isLeft):
        mono = pipeline.createMonoCamera()
        mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        if isLeft:
            mono.setBoardSocket(dai.CameraBoardSocket.LEFT)
        else:
            mono.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        return mono
    

 