import cv2
import pathlib
from deepface import DeepFace

cascadepath=pathlib.Path(cv2.__file__).parent.absolute() /"data/haarcascade_frontalface_default.xml"


clf=cv2.CascadeClassifier(str(cascadepath))

cam=cv2.VideoCapture(0)

while True:
    ret, frame=cam.read()
    gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb=cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)

    faces=clf.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=8, minSize=(30, 30))

    for (x,y,w,h) in faces:
        face= rgb[y:y+h,x:x+w]

        details=DeepFace.analyze(face, actions=['age','emotion','race'], enforce_detection=False)
        resulte=details[0]['dominant_emotion']
        resulta=details[0]['age']
        resultr=details[0]['dominant_race']
        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame,f"Emotion:{resulte}",(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.putText(frame,f"Age:{str(resulta)}",(x,y-40),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)
        cv2.putText(frame,f"Race:{resultr}",(x,y-70),cv2.FONT_HERSHEY_SIMPLEX,1,(255,0,0),2,cv2.LINE_AA)


    cv2.imshow("Face",frame)
    if cv2.waitKey(2) == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
