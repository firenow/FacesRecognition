#!/usr/bin/env python

# Import the required modules
import cv2, os
import numpy as np
from PIL import Image
import time


class FaceRecognizator:
    face_cascade = None
    recognizer = None
    img_path = None
    persons = ()

    def __init__(self):
        self.img_path = "./resources/yalefaces"

    def get_images_and_labels(self, path):
        # Append all the absolute image paths in a list image_paths
        # We will not read the image with the .sad extension in the training set
        # Rather, we will use them to test our accuracy of the training
        image_paths = [os.path.join(path, f) for f in os.listdir(path) if not f.endswith('.sad')]
        # images will contains face images
        images = []
        # labels will contains the label that is assigned to the image
        labels = []
        for image_path in image_paths:
            # Read the image and convert to grayscale
            image_pil = Image.open(image_path).convert('L')
            # Convert the image format into numpy array
            image = np.array(image_pil, 'uint8')
            # Get the label of the image
            nbr = int(os.path.split(image_path)[1].split(".")[0].replace("subject", ""))
            # Detect the face in the image
            faces = self.face_cascade.detectMultiScale(image)
            # If face is detected, append the face to images and the label to labels
            for (x, y, w, h) in faces:
                images.append(image[y: y + h, x: x + w])
                labels.append(nbr)
                cv2.imshow("Ajoute images au set de training...", image[y: y + h, x: x + w])
                cv2.waitKey(10)

        #time.sleep(1)
        # return the images list and labels listpath
        return images, labels

    # Call the get_images_and_labels function and get the face images and the
    # corresponding labels
    def init(self):
        print("Initialisation du modèle via les photos")
        # For face detection we will use the Haar Cascade provided by OpenCV.
        cascade_path = "./resources/haarcascade_frontalface_default.xml"
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        # For face recognition we will the the LBPH Face Recognizer
        self.recognizer = cv2.face.createLBPHFaceRecognizer()
        images, labels = self.get_images_and_labels(self.img_path)
        self.recognizer.train(images, np.array(labels))

    def label_new_faces(self):
        while True:
            for person in self.persons:
                if person.is_not_identified():
                    result = cv2.face.MinDistancePredictCollector()
                    self.recognizer.predict(person.img_gray[y: person.y + person.h, x: person.x + person.w], result, 0)
                    person.set_prediction_result(result)

    def label_new_face(self, person):
        print("Tentative d'identification ...")

        if not person.is_identified():
            result = cv2.face.MinDistancePredictCollector()
            self.recognizer.predict(person.get_face(), result, 0)
            person.set_prediction_result(result)
        else:
            print("Personne déjà identifiée")

    def recognize_faces(self):
        self.init()
        # self.testImgModel()
        self.label_new_faces()
