import cv2 as cv
import numpy as np
import time

if __name__ == "__main__":
    cap = cv.VideoCapture(0)
    
    # Cho camera khoi dong va can bang sang
    for i in range(50):
        cap.read()
        
    base_frame = None
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame is not None:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            gray = cv.GaussianBlur(gray, (25,25), 0)
        
        key = cv.waitKey(10) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            base_frame = None
            print("Da reset background")
            
        if base_frame is None:
            base_frame = gray
            continue

        delta = cv.absdiff(base_frame, gray)
        nguong = cv.threshold(delta, 25, 255, cv.THRESH_BINARY)[1]

        nguong = cv.dilate(nguong, None, iterations=2) 

        bien,_ = cv.findContours(nguong, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        for b in bien:
            if cv.contourArea(b) < 1000:
                continue
            (x,y,w,h) = cv.boundingRect(b)
            cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        
        cv.imshow("Webcam", frame)

        cv.imshow("Nguong", nguong)



    #cv.waitkey(0)
    cv.destroyAllWindows()