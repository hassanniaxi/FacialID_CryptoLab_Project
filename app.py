from flask import Flask, render_template, request, redirect, url_for, session, jsonify
import cv2
import os
import sqlite3
import hashlib
import json
import numpy as np
import base64
import hashlib
from os import listdir
from os.path import isfile, join
import time
from PIL import Image

app = Flask(__name__)


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
            return render_template('face_login.html', error_message1=error2)

        # Hash the password 
        hashed_password = hashlib.sha256(password.encode()).hexdigest()

        # Store user data in the database
        conn = sqlite3.connect('users_database.db')
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users_database (name, email, password) VALUES (?, ?, ?)", (name, email, hashed_password))
        conn.commit()
        curr_user = name
        conn.close()
        
        return render_template('logged.html', user_name=curr_user)

    return render_template('face_login.html')

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
             return render_template('logged.html', user_name=curr_user)
          else:
             conn.close()
             error = "Password is incorrect!"
             return render_template('index.html', error_message=error)
        else:
            conn.close()
            error = "User doesn't exist!"
            return render_template('index.html', error_message=error)

    return render_template('face_login.html')


@app.route('/logout')
def logout():
    session.pop('email', None)
    return redirect(url_for('index'))



face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + '/haarcascade_frontalface_default.xml')
def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.5, 7)
    
    if len(faces) == 0:
     return None

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cropped_face = img[y:y+h, x:x+w]
        cropped_face = cv2.resize(cropped_face, (200, 200))
    return cropped_face 


@app.route('/register_face', methods=['POST'])
def recognize_face():
    name_registered = request.form['Name']
    cap = cv2.VideoCapture(0)
    img_id = 0
    Databasee = 'database/'
    pics =0 
    while True:
        ret, frame = cap.read() 
        face = face_extractor(frame)
        if face is not None:
            img_id += 1
            if not os.path.exists(Databasee):
                os.makedirs(Databasee)
            
            face_database_path = os.path.join(Databasee, f"{name_registered}.{img_id}.jpg")
            if os.path.isfile(face_database_path):
                cv2.putText(frame, "Face already exists!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                return jsonify({'message': "Face already exists!"})
            else:
                cv2.imwrite(face_database_path, face)
                cv2.putText(frame, "Face Registering..", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                pics+=1
        else: 
            cv2.putText(frame, "No face found", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
 
        if pics == 200:
            cap.release()
            cv2.destroyAllWindows()
            return jsonify({'message': "Face Registered", 'face_user': name_registered})
             
        cv2.imshow('Face Detector', frame)
 
        if cv2.waitKey(1) == 13: # Check for 'Enter' key press to exit
            break

    cap.release()
    cv2.destroyAllWindows()
 
@app.route('/redirecttt/<user_name>')
def redirect_page(user_name):
    return render_template('redirect.html', user_name=user_name)
    
@app.route('/logged/<user_name>')
def logged(user_name): 
    return render_template('logged.html', user_name=user_name)


 
data_path = 'database/'
path = [os.path.join(data_path, f) for f in os.listdir(data_path)]# if f.endswith('.jpg')]

def train_data_classifier():
    Training_Data, Labels = [], []

    for i, file in enumerate(os.listdir(data_path)):
        image_path = os.path.join(data_path, file)

        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None or img.size == 0:
                print(f"Error reading image: {image_path}")
                continue

            img = cv2.resize(img, (200, 200))
            Training_Data.append(np.asarray(img, dtype=np.uint8))
            Labels.append(i)
        except Exception as e:
            print(f"Exception occurred: {str(e)}")

    if not Training_Data or not Labels:
        print("No valid training data found.")
        return None

    Labels = np.asarray(Labels, dtype=np.int32)

    model = cv2.face.LBPHFaceRecognizer_create()
    model.train(np.asarray(Training_Data), np.asarray(Labels))

    return model

# model = train_data_classifier()


face_classifier2 = cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_default.xml")

def face_detector(img, size=0.5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_classifier2.detectMultiScale(gray, 1.5, 7)

    if len(faces) == 0:
        return img, None

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi = img[y:y + h, x:x + w]
        roi = cv2.resize(roi, (200, 200))

    return img, roi

@app.route('/loginByFace')
def faceLogin():
    #model = train_data_classifier()
    model = train_data_classifier()
    cap = cv2.VideoCapture(0)
    count = 0
    while True:
        ret, frame = cap.read()
        image, face = face_detector(frame)
        
        if face is not None:
            try:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                result = model.predict(face)

                if result[1] < 60:  
                    P_name_with_extension = os.path.basename(path[0])
                    # Split the filename by dots and take the first part
                    P_name_parts = P_name_with_extension.split('.')
                    P_name = P_name_parts[0]
                    print("Recognized:", P_name)  
                    cv2.putText(image, P_name, (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                    count+=1
                    if count == 5:
                        cap.release()
                        cv2.destroyAllWindows()
                        return jsonify({'message': "Login Successfully!!!", 'face_user': P_name})
                else:
                    cv2.putText(image, "Unknown Face", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
            except Exception as e:
                cv2.putText(image, "Face Not Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)
                pass
        else:
            cv2.putText(image, "No Face Found", (250, 450), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Face Decetor', image)

        if cv2.waitKey(1) == 13:
            cap.release()
            cv2.destroyAllWindows()
            return jsonify({'message': "Login Failed. Face not recognized."})
            break

    cap.release() 
    cv2.destroyAllWindows()

if __name__ == '__main__':
    init_db()
    app.run(debug=True)