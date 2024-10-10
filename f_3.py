import face_recognition
import cv2

# Load images for recognition
known_image = face_recognition.load_image_file("known_person.jpg")
unknown_image = face_recognition.load_image_file("unknown_person.jpg")

# Encode faces (get facial features for recognition)
known_encoding = face_recognition.face_encodings(known_image)[0]
unknown_encoding = face_recognition.face_encodings(unknown_image)[0]

# Compare faces and get the result
results = face_recognition.compare_faces([known_encoding], unknown_encoding)

if results[0]:
    print("Match found!")
else:
    print("No match found.")
