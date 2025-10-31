import cv2
import pathlib
import time
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db
import tkinter as tk
from PIL import Image, ImageTk

# ---------------------------
# Firebase setup
# ---------------------------
cred = credentials.Certificate("reconhecimentofacial-463e1-firebase-adminsdk-fbsvc-284effdfd3.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "https://reconhecimentofacial-463e1-default-rtdb.firebaseio.com/face_detections/-OZKysFLDkREkzvifgp7"
})
ref = db.reference("/face_detections")

# ---------------------------
# OpenCV face detection setup
# ---------------------------
cascade_path = pathlib.Path(cv2.__file__).parent / "data/haarcascade_frontalface_default.xml"
clf = cv2.CascadeClassifier(str(cascade_path))

camera = cv2.VideoCapture(0)

# Variables for 5-second threshold
face_detected_start = None
threshold_seconds = 5
confirmed_faces_count = 0
last_detection_time = "N/A"

# ---------------------------
# Tkinter GUI setup
# ---------------------------
window = tk.Tk()
window.title("Painel De Controle")

# Labels
counter_label = tk.Label(window, text=f"Faces Detectadas: {confirmed_faces_count}", font=("Arial", 14))
counter_label.pack()


time_label = tk.Label(window, text=f"Última Detecção: {last_detection_time}", font=("Arial", 14))
time_label.pack()

# Canvas for video
video_label = tk.Label(window)
video_label.pack()

# ---------------------------
# Update function
# ---------------------------
def update_frame():
    global face_detected_start, confirmed_faces_count, last_detection_time

    ret, frame = camera.read()
    if ret:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = clf.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw rectangles
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # 5-second detection logic
        if len(faces) > 0:
            if face_detected_start is None:
                face_detected_start = time.time()
            else:
                elapsed = time.time() - face_detected_start
                if elapsed >= threshold_seconds:
                    confirmed_faces_count += 1
                    last_detection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    print(f"Face confirmed at {last_detection_time}!")

                    # Push to Firebase
                    ref.push({
                        "timestamp": last_detection_time,
                        "status": "confirmed"
                    })

                    # Reset to avoid double counting
                    face_detected_start = None
        else:
            face_detected_start = None

        # Update GUI labels
        counter_label.config(text=f"Faces Detectadas: {confirmed_faces_count}")
        time_label.config(text=f"Última Detecção: {last_detection_time}")

        # Convert frame for Tkinter
        cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        video_label.imgtk = imgtk
        video_label.configure(image=imgtk)

    # Schedule next frame update
    window.after(10, update_frame)

# Start updating frames
update_frame()
window.mainloop()

# ---------------------------
# Cleanup
# ---------------------------
camera.release()
cv2.destroyAllWindows()
