import cv2
import os
import numpy as np
face_database_path = 'D:/LECTURES/SEMESTER-5/Security and Forensic/Project/FacialID_CryptoLab_Project/database'
face_cascade_path = os.path.join(cv2.data.haarcascades, 'D:/LECTURES/SEMESTER-5/Security and Forensic/Project/FacialID_CryptoLab_Project/haarcascade_frontalface_default.xml')

# Initialize CascadeClassifier with the correct path
face_classifier = cv2.CascadeClassifier(face_cascade_path)
files_in_faceDB = [f for f in os.listdir(face_database_path) if f.endswith('.jpg')]

if face_classifier.empty():
    print("Error: Unable to load the Haar cascade classifier.")
    exit()
name = input("Enter Name: ")

name_registered =name

def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return None

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cropped_face = img[y:y + h, x:x + w]
        cropped_face = cv2.resize(cropped_face, (200, 200))
    return cropped_face

def face_detector(img, size=0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    if len(faces) == 0:
        return img, []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = img[y:y + h, x:x + w]
        roi = cv2.resize(roi, (200, 200))

    return img, roi

def train_data_classifier():
    Training_Data, Labels = [], []

    for i, file in enumerate(files_in_faceDB):
        image_path = os.path.join(face_database_path, file)
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) 
        img = cv2.resize(img, (200, 200))  # Resize all images to a consistent size
        Training_Data.append(np.asarray(img, dtype=np.uint8))
        Labels.append(i)

    Labels = np.asarray(Labels, dtype=np.int32) 

    m = cv2.face.LBPHFaceRecognizer_create()
    m.train(np.asarray(Training_Data), np.asarray(Labels))
    print("Dataset Model Training Complete!!!!!")
    return m


cap = cv2.VideoCapture(0)
model2 = train_data_classifier()
pics = 1
i=0
while True:
    ret, frame = cap.read()
    image, face = face_detector(frame)
    if face_extractor(frame) is not None:
        if not os.path.exists(face_database_path):
            os.makedirs(face_database_path)

        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        result = model2.predict(face)

        if result[1] < 1000:
            confidence = int(100 * (1 - (result[1]) / 300))
            print(confidence)
            i=confidence

        if confidence > 80:
            i = (i % len(files_in_faceDB))
            print(i)
            P_name_with_extension = os.path.basename(files_in_faceDB[i])
            P_name_parts = P_name_with_extension.split('.')
            P_name = P_name_parts[0]
            cv2.putText(frame, "Face already exists!", (250, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cap.release()
            cv2.destroyAllWindows()
            print("Face already exist as: " + P_name)
            break
        else:
            file_name = f"{name_registered}.{pics}.jpg"
            face_database = os.path.join('database', file_name)
            cv2.imwrite(face_database, face)
            cv2.putText(frame, "Face Registering..", (250, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            pics += 1
    else:
        cv2.putText(frame, "No face found", (250, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        pass

    if pics == 10:
        cap.release()
        cv2.destroyAllWindows()
        print(name_registered)
        print("Face Registered")
        break

    cv2.imshow('Face Detector', frame)

    if cv2.waitKey(1) == 13:  # Check for 'Enter' key press to exit
        cap.release()
        cv2.destroyAllWindows()
        print("message Dataset does'nt Fully Completed")
        break

cap.release()
cv2.destroyAllWindows()