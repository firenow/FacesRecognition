#!/usr/bin/env python

import numpy as np
import cv2
from persons import Persons
import time

'''
Recognize new members on the video-stream
'''


class FaceDetector:
    def __init__(self, persons):
        self.face_cascade = cv2.CascadeClassifier('./resources/haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('./resources/haarcascade_eye.xml')
        self.persons = persons

    def get_video_stream(self):
        return cv2.VideoCapture(0)

    def find_all_faces(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        # ajouter une détection des yeux et vérifier que les yeux sont compris dans la tete

        return faces

    def person_corresponds(self, x, y):
        for person in self.persons.persons:
            if person.corresponds(x, y, 100):
                #print("Personn")
                return True

        return False

    def person_corresponds(self, x, y, w, h):
        for person in self.persons.persons:
            if person.corresponds(x, y, w, h, 100):
                #print("Personn")
                return True

        return False

    def get_person_corresponds(self, x, y, w, h):
        for person in self.persons.persons:
            if person.corresponds(x, y, w, h, 100):
                #print("Personn")
                return person

    def filter_old_faces(self, all_faces):
        new_faces = []

        for face in all_faces:
            if (not self.person_corresponds(face[0], face[1], face[2], face[3])):
                #print("New person detected")
                new_faces.append(face)

        return new_faces

    def filter_new_faces(self, all_faces):
        old_faces = []

        for face in all_faces:
            if (self.person_corresponds(face[0], face[1], face[2], face[3])):
                #print("New person detected")
                old_faces.append(face)

        return old_faces

    def store_info_persons(self, old_faces):
        for face in old_faces:
            person = self.get_person_corresponds(face[0], face[1], face[2], face[3])

            if (person is not None):
                person.add_position(face[0], face[1], face[2], face[3])

    def init_new_persons(self, img, new_faces):
        for new_face in new_faces:
            self.persons.add_person(img, new_face[0], new_face[1], new_face[2], new_face[3])

    def launch_videofaces_prediction(self, img):
        all_faces = self.find_all_faces(img)
        new_faces = self.filter_old_faces(all_faces)
        old_faces = self.filter_new_faces(all_faces)

        if (len(new_faces) >0 ):
            print(" /!\%d nouvelle personne\n" % (len(new_faces)))

        print("%d total personnes détectées\n" % (len(all_faces)))
        self.init_new_persons(img, new_faces)

        self.store_info_persons(old_faces)

        self.persons.paint(img)
        self.persons.remove_unfollowed_persons()


        # temporise
        k = cv2.waitKey(30) & 0xff

    def predict_faces(self):
        videostream = self.get_video_stream()

        while True:
            try:
                ret, img = videostream.read()
                self.launch_videofaces_prediction(img)
                cv2.imshow('img', img)
                k = cv2.waitKey(30) & 0xff
                time.sleep(0.05)

            except KeyboardInterrupt:
                videostream.release()
                cv2.destroyAllWindows()
