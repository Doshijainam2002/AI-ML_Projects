import cv2 as cv
import numpy as np
import csv
from datetime import datetime
import os
from skimage.metrics import structural_similarity as compare_ssim
import time

capture = cv.VideoCapture(0)

# Load pre-trained Haarcascades face detection model
face_cascade = cv.CascadeClassifier(cv.data.haarcascades + 'haarcascade_frontalface_default.xml')

image_dir = "/Users/jainamdoshi/Desktop/Machine Learning Models/Projects/FacialAttendanceSystem/Batch/PCM"

known_face_encodings = []
known_face_names = []

for file_name in os.listdir(image_dir):
    if file_name.endswith(".jpg"):
        image_path = os.path.join(image_dir, file_name)
        image = cv.imread(image_path)

        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:
            for (x, y, w, h) in faces:
                face = gray[y:y + h, x:x + w]
                face_resized = cv.resize(face, (100, 100))  # Resize to a common size for comparison
                known_face_encodings.append(face_resized)
                known_face_names.append(os.path.splitext(file_name)[0])

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(current_date + '.csv', "w+", newline='')
lnwriter = csv.writer(f)

entry_delay = 30  # Set the desired delay in seconds
last_entry_times = {}  # To keep track of last entry times for each person

# Load existing attendance records from the CSV file
attendance_records = set()
if os.path.exists(current_date + '.csv'):
    with open(current_date + '.csv', "r") as f:
        csv_reader = csv.reader(f)
        for row in csv_reader:
            attendance_records.add(row[0])

while True:
    _, frame = capture.read()
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = gray_frame[y:y + h, x:x + w]
        face_resized = cv.resize(face, (100, 100))  # Resize to a common size for comparison

        # Compare the detected face with known face encodings using SSI
        best_match_similarity = 0
        best_match_index = -1

        for idx, known_encoding in enumerate(known_face_encodings):
            similarity = compare_ssim(face_resized, known_encoding)
            if similarity > best_match_similarity:
                best_match_similarity = similarity
                best_match_index = idx

        if best_match_index != -1:
            name = known_face_names[best_match_index]
            current_time = now.strftime("%H-%M-%S")

            # Check if the person is not already marked as present and if enough time has passed since the last entry
            if name not in attendance_records and (name not in last_entry_times or (time.time() - last_entry_times[name]) > entry_delay):
                last_entry_times[name] = time.time()
                lnwriter.writerow([name, current_time])
                attendance_records.add(name)  # Mark the person as present

            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv.putText(frame, name, (x, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv.imshow("Attendance System", frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv.destroyAllWindows()
f.close()
