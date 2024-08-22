# Made by CodeHack

import cv2

# Face, nose, and mouth classifier
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
nose_detector = cv2.CascadeClassifier('Nariz.xml')
mouth_detector = cv2.CascadeClassifier('Mouth.xml')

# Choose webcam to detect nose and mouth in
webcam = cv2.VideoCapture(0)

# Iterate over frames
while True:
    # Read frames
    successful_frame_read, frame = webcam.read()

    # Must convert to grayscale
    grayscale_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

    # Detect faces
    faces = face_detector.detectMultiScale(grayscale_frame)

    # Find the faces
    for (x, y, w, h) in faces:
        # Draw rectangle around the faces
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 200, 50), 4)

        # Get the faces
        the_face = frame[y:y + h, x:x + w]
        # Convert to grayscale
        grayscale_face = cv2.cvtColor(the_face, cv2.COLOR_RGB2GRAY)

        # Detect nose
        nose = nose_detector.detectMultiScale(grayscale_face, scaleFactor=1.2, minNeighbors=8)

        # Detect mouth
        mouth = mouth_detector.detectMultiScale(grayscale_face, scaleFactor=1.7, minNeighbors=30)

        # Find nose in faces
        for (x_, y_, w_, h_) in nose:
            # Draw rectangle around nose
            cv2.rectangle(the_face, (x_, y_), (x_ + w_, y_ + h_), (200, 50, 50), 4)

        # Find mouth in faces
        for (x_, y_, w_, h_) in mouth:
            # Draw rectangle around mouth
            cv2.rectangle(the_face, (x_, y_), (x_ + w_, y_ + h_), (50, 50, 200), 4)

    # Display frames with face, nose and mouth
    cv2.imshow('CodeHack Face Mask Detector', frame)
    key = cv2.waitKey(1)

    # Stop if Q key is pressed
    if key == 81 or key == 113:
        break

# Cleanup
webcam.release()
cv2.destroyAllWindows()

print("Code Completed")

