import numpy as np
import cv2
import sys

def procEyes(mat):
    mat = cv2.medianBlur(mat,11) 
    cv2.imshow("hey",mat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  
    mat_gry = cv2.cvtColor(mat, cv2.cv.CV_BGR2GRAY)
    cv2.equalizeHist( mat_gry, mat_gry ) 
    x,mat_gry = cv2.threshold( mat_gry , 25, 255, cv2.THRESH_BINARY )
    cv2.imshow("hey",mat_gry)
    cv2.waitKey(0)
    cv2.destroyAllWindows()  


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')




#Ubicacao completa da imagen
name = sys.argv[1]
img = cv2.imread(name) 
rows,cols = img.shape[:2]

flag =False




for i in range(12):
    if flag:
        break
    M = cv2.getRotationMatrix2D((cols/2,rows/2),30*i,1) #Fazer rotacoes
    dst = cv2.warpAffine(img,M,(cols,rows))
    gray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)

    minTam=(img.shape[0]/5,img.shape[1]/5 ) #tamanho minimo para ser um rosto
    
    faces = face_cascade.detectMultiScale(gray,1.1,3,minSize=minTam)

    for (x,y,w,h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = dst[y:y+h, x:x+w]
        
        minTamOlhos=(roi_color.shape[0]/6,roi_color.shape[1]/6 )
        eyes = eye_cascade.detectMultiScale(roi_gray,1.1,8,minSize=minTamOlhos)
        if(eyes==None or len(eyes)==0): #Se nao ha olhos nao e rosto
            continue 
        img2 = cv2.rectangle(dst,(x,y),(x+w,y+h),(255,0,0),5)        
        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),5)    
            procEyes(roi_color[ey+eh/5:ey+4*eh/5,ex+ew/5:ex+4*ew/5])        
        M = cv2.getRotationMatrix2D((cols/2,rows/2),-30*i,1)
        dst = cv2.warpAffine(dst,M,(cols,rows))                    
        
        cv2.imwrite(sys.argv[2],dst)
        flag=True        
        
        
        
        
