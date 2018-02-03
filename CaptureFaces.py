import cv2
import shutil
import os
import pathlib
import csv

cam = cv2.VideoCapture(0)

cv2.namedWindow("test")

img_counter = 0
person_name = "Niranjay"
pathlib.Path('train/'+person_name).mkdir(parents=True, exist_ok=True)
pathlib.Path('test/'+person_name).mkdir(parents=True, exist_ok=True)  
image_names = []
while True:
    ret, frame = cam.read()
    cv2.imshow("test", frame)
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed

        img_name = ("opencv_frame_{}.jpg".format(img_counter))           
        cv2.imwrite(img_name, frame)
        if img_counter%4==0:
            shutil.move(img_name,'test/'+person_name+'/'+img_name)
        else:
            shutil.move(img_name,'train/'+person_name+'/'+img_name)
        print("{} written!".format(img_name))
        img_counter += 1
  
cam.release()

cv2.destroyAllWindows()