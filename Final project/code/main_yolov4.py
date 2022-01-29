#!/usr/bin/python3

import os
import sys
from time import sleep
import time
from models import Yolov4
import cv2


# build yolo model
model = Yolov4(class_name_path="./class_names/coco_classes.txt",
               weight_path="./yolov4.weights")



def main():
    cap = cv2.VideoCapture("../video/person_bike.mp4")
    input_fps = cap.get(cv2.CAP_PROP_FPS)
    out = cv2.VideoWriter("../output/person_bike_310515010.avi",cv2.VideoWriter_fourcc('X','V','I','D'), input_fps, (1280, 720))

    while (cap.isOpened()):
        ret, frame = cap.read()

        if ret == True:
            start = time.time()
            output_img, detections = model.predict_img(frame, random_color=False, plot_img=False, return_output=True)     
            end = time.time()
            print("predict time：%.2f 秒" % (end - start))
            #print("class_name :", detections['class_name'])
            cv2.imshow("output_img", output_img)
            #out.write(output_img) 

            if cv2.waitKey(1) & 0xFF == 27:
                break
    print("================= finish prediction =================")
    cap.release()
    cv2.destroyAllWindows()

# main 
if __name__ == "__main__":
    main()