# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 19:20:32 2020

@author: Manikant
"""

import cv2
face_cascade=cv2.CascadeClassifier('F:\P23-Installations\Installations\haarcascade_frontalface_default.xml')
eye_cascade=cv2.CascadeClassifier('F:\P23-Installations\Installations\haarcascade_eye.xml')
smile_cascade=cv2.CascadeClassifier('F:\P23-Installations\Installations\haarcascade_smile.xml')
def detect(gray,frame):
    faces=face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_color=frame[y:y+h,x:x+w]
        eyes=eye_cascade.detectMultiScale(roi_gray,1.2,18)
        for (ex,ey,eh,ew) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)
        smiles=smile_cascade.detectMultiScale(roi_gray,1.7,20)
        for (sx,sy,sh,sw) in smiles:
            cv2.rectangle(roi_color,(sx,sy),(sx+sw,sy+sh),(255,0,255),2)
    return frame

video_capture= cv2.VideoCapture(0)
while True:
    _,frame= video_capture.read()
    gray= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    canvas= detect(gray,frame)
    cv2.imshow('Video',canvas)
    if cv2.waitKey(1) & 0xFF==ord('q'):
        break
video_capture.release()
cv2.destroyAllWindows()