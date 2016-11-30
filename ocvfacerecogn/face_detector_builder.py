#!/usr/bin/env python

from face_recognizator import FaceRecognizator
from face_detector import FaceDetector
from persons import Persons


def build_model():
    print("Initialisation...")
    face_recognizator = FaceRecognizator()
    face_recognizator.init()
    persons = Persons(face_recognizator)
    print("\n\nDémarrage...")
    face_detector = FaceDetector(persons)

    return face_detector
