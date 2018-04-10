import cv2

# Carico i cascade per il riconoscimento del volto e degli occhi
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
face_profile_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye_tree_eyeglasses.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# Sorgente video
cap = cv2.VideoCapture(0)

while(True):
    # Catturo frame per frame
    ret, img = cap.read()
    # Trasformo il frame catturato in scala di grigi
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Catturo le facce presenti nel frame "grigio"
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    faces_profile = face_profile_cascade.detectMultiScale(gray, 1.3, 5)

    for(x,y,w,h) in faces:
        # Per ogni faccia disegno un rettangolo intorno
        cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        # Catturo gli occhi presenti nella faccia catturata
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for(ex, ey, ew, eh) in eyes:
            # Per ogni occhio disegno un rettangolo attorno
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

        # Catturo lo "smile"
        """smile = smile_cascade.detectMultiScale(roi_gray)
        for (sx, sy, sw, sh) in smile:
            cv2.rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 0, 255), 2)"""
            
    for(x,y,w,h) in faces_profile:
        # Per ogni faccia disegno un rettangolo intorno
        cv2.rectangle(img, (x,y), (x+w, y+h), (0, 0, 255), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = img[y:y+h, x:x+w]

        # Catturo gli occhi presenti nella faccia catturata
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for(ex, ey, ew, eh) in eyes:
            # Per ogni occhio disegno un rettangolo attorno
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    # Mostro il tutto
    cv2.imshow('Face and eyes detection', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Attendo che venga premuto il tasto "Q" per uscire
        break

# Distruggo gli oggetti creati
cap.release()
cv2.destroyAllWindows()
	    
