# FacialID_CryptoLab_Project


**Welcome to FacialID CryptoLab**

The "FacialID CryptoLab" appears to be a Flask-based application leveraging facial recognition for user authentication and registration. Here's a brief breakdown of its functionalities:

1. User Registration/Login:
   - Users can register by providing their name, email, and password.
   - Passwords are hashed (using SHA256) before being stored in the database for security.
   - Existing user checks are implemented during registration and login.

2. Facial Registration:
   - Users can register their faces by capturing images through the webcam.
   - The application detects faces using the OpenCV library.
   - Captured facial images are stored in a 'database/' directory, named after the registered user.

3. Facial Authentication/Login:
   - The system allows users to log in using facial recognition.
   - The facial recognition model is trained using the images stored during registration.
   - Upon login, the webcam captures the face, and the model recognizes it against the stored database.

4. Session Management:
   - Flask sessions are used to maintain user login state.

5. **Database:**
   - SQLite is used as the database to store user information and registered facial images.

6. Web Interface:
   - HTML templates are used for user interaction and display.

7. Error Handling:
   - Error messages are displayed for incorrect login attempts, password mismatches, or existing users during registration.

8. Security Measures:
   - A secure secret key is generated for Flask session management.
   - Passwords are hashed before storing to enhance security.

This application integrates user authentication via both traditional (email/password) and facial recognition methods, offering a multi-factor authentication approach for user access.
