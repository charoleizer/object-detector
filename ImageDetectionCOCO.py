from imageai.Detection import ObjectDetection, VideoObjectDetection
import os
import cv2
import os

clear = lambda: os.system('clear')
clear()

execution_path = os.getcwd()

objectDetector = ObjectDetection()
objectDetector.setModelTypeAsRetinaNet()
objectDetector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
objectDetector.loadModel()

cam = cv2.VideoCapture(0)

camON, frame = cam.read()           
while camON:
    key = cv2.waitKey(20)
    
    if key == 27:    
        break
    
    rervalt, frame = cam.read()
    cv2.imshow('Window', frame)

    if key == 32:
        clear()

        cv2.imwrite('current-object.png', frame)
        objectDetections = objectDetector.detectObjectsFromImage(input_image=os.path.join(execution_path , "current-object.png"), output_image_path=os.path.join(execution_path , "current-object-analysis.png"))

        for eachObject in objectDetections:
            print(eachObject["name"] , " : " , eachObject["percentage_probability"] )

        analysedFrame = cv2.imread('current-object-analysis.png')
        cv2.imshow('Window', analysedFrame)

        while (True):
            key = cv2.waitKey(20)
            if key == 27:    
                break


cam.release()
cv2.destroyAllWindows()