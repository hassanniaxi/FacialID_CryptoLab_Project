import cv2
import numpy as np
import os

# Define paths
face_database_path = 'D:/LECTURES/SEMESTER-5/Security and Forensic/Project/FacialID_CryptoLab_Project/database'
face_cascade_path = os.path.join(cv2.data.haarcascades, 'haarcascade_frontalface_default.xml')

# Load Haar cascade classifier
face_classifier = cv2.CascadeClassifier(face_cascade_path)

# List of face image files in the database
files_in_faceDB = [f for f in os.listdir(face_database_path) if f.endswith('.jpg')]

def train_data_classifier():
    # Function to train the face recognition model
    Training_Data, Labels = [], []

    for i, file in enumerate(files_in_faceDB):
        image_path = os.path.join(face_database_path, file)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (200, 200))
        Training_Data.append(np.asarray(img, dtype=np.uint8))
        Labels.append(i)

    Labels = np.asarray(Labels, dtype=np.int32)

    m = cv2.face.LBPHFaceRecognizer_create()
    m.train(np.asarray(Training_Data), np.asarray(Labels))
    print("Dataset Model Training Complete!!!!!")
    return m

def face_detector(img, size=0.5):
    # Function to detect faces in an image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return img, []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = img[y:y + h, x:x + w]
        roi = cv2.resize(roi, (200, 200))

    return img, faces

cap = cv2.VideoCapture(0)  
count = 0
model = train_data_classifier()
i = 0
while True:
    ret, frame = cap.read()
    img, faces = face_detector(frame)  # faces is a list of faces
    for (x, y, w, h) in faces:
        try:
            face = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)  # Convert each face to grayscale
            result = model.predict(face)

            if result[1] < 100:
                confidence = int(100 * (1 - (result[1]) / 300))
                print(confidence)
                i=confidence
            # else:
            #     confidence = 0

            if confidence > 65:
                i =  i = (i % len(files_in_faceDB))  # Assuming you have only one result
                P_name_with_extension = os.path.basename(files_in_faceDB[result[i]])
                P_name_parts = P_name_with_extension.split('.')
                P_name = P_name_parts[0]
                cv2.putText(img, P_name, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                cv2.imshow('Face Cropper', img)
                count += 1

                if count == 100:
                    print(P_name)
                    print("Login Successfully!!!")
                    break
            else:
                cv2.putText(img, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
                cv2.imshow('Face Cropper', img)
        except Exception as e:
            cv2.putText(img, "Face Not Found", (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
            cv2.imshow('Face Cropper', img)
            pass

    cv2.imshow('Face Cropper', img)
    if cv2.waitKey(1) == 13:
        print("Login Failed. Face not recognized.")
        break

# Release resources outside the loop
cap.release()
cv2.destroyAllWindows()
