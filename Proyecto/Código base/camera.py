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
    
    def createColorCamera(self, previewSize = (300,300)):
        pipeline = dai.Pipeline()
        cam_rgb = pipeline.create(dai.node.ColorCamera)
        cam_rgb.setPreviewSize(previewSize)
        cam_rgb.setInterleaved(False)

        xout_rgb = pipeline.create(dai.node.XLinkOut)
        xout_rgb.setStreamName("rgb")
        cam_rgb.preview.link(xout_rgb.input)

        return dai.Device(pipeline, usb2Mode=self.usb2mode)
    
    def createStereoDepth(self, threshold):
        pipeline = dai.Pipeline()
        stereo_depth = pipeline.create(dai.node.StereoDepth)
        stereo_depth.initialConfig.setConfidenceThreshold(threshold)

    def getFrame(self, queue):
        frame = queue.get()
        return frame.getCvFrame()

    def getMonoCamera(self, pipeline, isLeft):
        mono = pipeline.createMonoCamera()
        mono.setResolution(dai.MonoCameraProperties.SensorResolution.THE_800_P)
        if isLeft:
            mono.setBoardSocket(dai.CameraBoardSocket.LEFT)
        else:
            mono.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        return mono
    

 