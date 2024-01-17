from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import cv2
import os
import sqlite3
import numpy as np
from os import listdir
from os.path import isfile, join
import hashlib

app = Flask(__name__)

app.secret_key = os.urandom(24)  # Generating a secure secret key


# app.config['SECRET_KEY'] = 'your_secret_key'


@app.route('/')
def index():
    return render_template('index.html')

app.secret_key = os.urandom(24)  # Generating a secure secret key
def init_db():
    conn = sqlite3.connect('users_database.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users_database (
            id INTEGER PRIMARY KEY AUTOINCREMENT, 
            name TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form['Name1']
        email = request.form['Email']
        password = request.form['Password']
        confirm_password = request.form['ConfirmPassword']

        # Check if passwords match
        if password != confirm_password:
            error1="Passwords do not match!"
            return render_template('face_login.html', error_message2=error1)

        # Check if email already exists in the database
        conn = sqlite3.connect('users_database.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users_database WHERE email=?", (email,))
        user = cursor.fetchone()
        conn.close()

        if user:
            error2="Email already registered!"
            return render_template('homePage.html', error_message1=error2)

        # Hash the password 
        hashed_password = hashlib.sha256(password.encode()).hexdigest()

        # Store user data in the database
        conn = sqlite3.connect('users_database.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users_database (name, email, password) VALUES (?, ?, ?)", (name, email, hashed_password))
        conn.commit()
        curr_user = name
        conn.close()

        return render_template('homePage.html', user_name=curr_user)

    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['Email']
        password = request.form['Password']

        # Retrieve user from the database
        conn = sqlite3.connect('users_database.db')
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM users_database WHERE email=?", (email,))
        user = cursor.fetchone()


        if user:
          if user[3] == hashlib.sha256(password.encode()).hexdigest():
             session['email'] = email  # Store user's email in session
             curr_user = user[1]  # Fetch the user's name (assuming it's at index 1)
             conn.close()
             return render_template('homePage.html', user_name=curr_user)
          else:
             conn.close()
             error = "Password is incorrect!"
             return render_template('index.html', error_message=error)
        else:
            conn.close()
            error = "User doesn't exist!"
            return render_template('index.html', error_message=error)

    return render_template('homePage.html')


@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect(url_for('index'))



# Define paths
face_database_path = 'database'
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
    return m

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
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) == 0:
        return img, []

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = img[y:y + h, x:x + w]
        roi = cv2.resize(roi, (200, 200))

    return img, faces


@app.route('/register_face', methods=['POST'])
def recognize_face():
    name_registered = request.form['Name']
    cap = cv2.VideoCapture(0)
    model = train_data_classifier()
    pics = 1
    while True:
        ret, frame = cap.read()
        image, face = face_detector(frame)
 
        if face_extractor(frame) is not None:
            if not os.path.exists(face_database_path):
                os.makedirs(face_database_path)

            # face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) 
            face = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            result = model.predict(face)

            if result[1] < 100:
                confidence = int(100 * (1 - (result[1]) / 300))
                i=confidence

            if confidence > 90:
                i = (i % len(files_in_faceDB))
                cv2.putText(frame, "Face already exists!", (250, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cap.release()
                cv2.destroyAllWindows()
                P_name_with_extension = files_in_faceDB[i]
                P_name_parts = P_name_with_extension.split('.')
                P_name = P_name_parts[0]

                return jsonify({'message': "Face already exist as: " + P_name})
            else:
                file_name = f"{name_registered}.{pics}.jpg"
                face_database = os.path.join('database', file_name)
                cv2.imwrite(face_database, face)
                cv2.putText(frame, "Face Registering..", (250, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                pics += 1
        else:
            cv2.putText(frame, "No face found", (250, 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            pass

        if pics == 50:
            cap.release()
            cv2.destroyAllWindows()
            session['_name'] = name_registered
            return jsonify({'message': "Face Registered"})

        cv2.imshow('Face Detector', frame)

        if cv2.waitKey(1) == 13:  # Check for 'Enter' key press to exit
            cap.release()
            cv2.destroyAllWindows()
            return jsonify({'message': "Dataset does'nt Fully Completed"})

    cap.release()
    cv2.destroyAllWindows()

@app.route('/redir')
def redirect_page():
    return render_template('redirectPage.html')

@app.route('/loginByFace')
def faceLogin():
    model = train_data_classifier()
    cap = cv2.VideoCapture(0)
    count = 0
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
                    i = confidence

                if confidence < 85:
                    i = (i % len(files_in_faceDB))
                    P_name_with_extension = files_in_faceDB[i]
                    P_name_parts = P_name_with_extension.split('.')
                    P_name = P_name_parts[0]
                    cv2.putText(img, P_name, (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 1)
                    cv2.imshow('Face Cropper', img)
                    count += 1
                    if count == 10:
                        cap.release()
                        cv2.destroyAllWindows()
                        session['_name'] = P_name
                        return jsonify({'message': "Login Successfully!!!"})
                else:
                    cv2.putText(img, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
                    cv2.imshow('Face Cropper', img)

            except Exception as e:
                cv2.putText(img, "Face Not Found", (x, y - 10), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 1)
                cv2.imshow('Face Cropper', img)
                pass
        cv2.imshow('Face Cropper', img)

        if cv2.waitKey(1) == 13:
            cap.release()
            cv2.destroyAllWindows()
            return jsonify({'message': "Login Failed. Face not recognized."})
            break

    cap.release() 
    cv2.destroyAllWindows()


@app.route('/homePage')
def logged(): 
    nname = session.get('_name') 
    return render_template('homePage.html', user_name=nname)

if __name__ == '__main__':
    init_db()
    app.run(debug=True)