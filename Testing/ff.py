import numpy as np
import cv2
import config
import glob
import os
import sys
import select
import time
face_cascade = cv2.CascadeClassifier(config.HAAR_FACES)
count = 0
camera = config.get_camera()
#image = cv2.imread('s1.jpg')
print 'Loading training data...'
model = cv2.createEigenFaceRecognizer()
model.load(config.TRAINING_FILE)
print 'Training data loaded!'

fi=0
while True:
    
    image = camera.read()
    cv2.imshow('ImageWindow',image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray,
                                        scaleFactor=config.HAAR_SCALE_FACTOR, 
                                        minNeighbors=config.HAAR_MIN_NEIGHBORS,
                                        minSize=config.HAAR_MIN_SIZE,
                                        flags = cv2.CASCADE_SCALE_IMAGE
                                        )
    print "Found {0} faces!".format(len(faces))
   
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        #print x,y,w,h
        #cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #cv2.imshow("Faces found" ,image)
        crop_height = int((config.FACE_HEIGHT / float(config.FACE_WIDTH)) * w)
        midy = y + h/2
        y1 = max(0, midy-crop_height/2)
        y2 = min(image.shape[0]-1, midy+crop_height/2)
        inn= image[y1:y2, x:x+w]
        inn = cv2.cvtColor(inn, cv2.COLOR_RGB2GRAY)
        inn=cv2.resize(inn, 
					  (config.FACE_WIDTH, config.FACE_HEIGHT), 
					  interpolation=cv2.INTER_LANCZOS4)
        label, confidence = model.predict(inn)
        label=str(label)
        print label
        #print 'Predicted {} face with confidence {} (lower is more confident).'.format(
					#'POSITIVE' if label == config.POSITIVE_LABEL else 'NEGATIVE', 
					#confidence)
        #if label == config.POSITIVE_LABEL and confidence < config.POSITIVE_THRESHOLD:
                
                
        if (label=='1' and confidence < 3000):
            print 'vaishnavi'
            cv2.putText(image,"vaishnavi", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
                 
        elif(label=='2' and confidence < 3000):
            
            print 'ashvini'
            cv2.putText(image,"ashvini", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
            
        elif(label=='3' and confidence < 3000):
            
            print 'sharayu'
            cv2.putText(image,"sharayu", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
           
        else:
            
            print 'unknown'
            cv2.putText(image,"unknown", (x,y), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 2)
         
        label=0          
        cv2.imshow('ImageWindow',image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break
 
        #time.sleep(0.2)
        filename = os.path.join('test/%03d.png' % count)
        cv2.imwrite(filename, inn)
        print 'Found face and wrote training image', filename
        count += 1

