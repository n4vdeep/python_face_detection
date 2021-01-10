import cv2
from random import randrange

# Load some pre-trained data on face frontals from opencv (haar cascade algorithm)
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Choose an image to detect faces in
img = cv2.imread('RDJ.jpg')

# Convert image to grayscale
grayscale_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect faces
face_coordinates = trained_face_data.detectMultiScale(grayscale_img)
# print face coordinates
print(face_coordinates)
for (x,y,w,h) in face_coordinates:
    # Draw rectangle around the faces
    cv2.rectangle(img, (x,  y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 5)


# Show image with opencv
cv2.imshow('Face Detection App', img)
cv2.waitKey()

print("Code Complete")
