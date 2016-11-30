#!/usr/bin/env python

from person import Person
from face_recognizator import FaceRecognizator
from face_follower import FaceFollower


class Persons:
    """A simple example class"""
    persons = []
    recognizator = None

    def __init__(self, face_recognizator):
        self.recognizator = face_recognizator

    def find_person_by_name(self, name):
        return next((x for x in self.persons if x.name == name), None)

    def add_information(self, name, x, y):
        person = self.find_person_by_name(name)

        if (person is None):
            person = Person(name, x, y)
            self.persons.append(person)
        else:
            person.addPosition(x, y)

        print("Personne ajoutée au modèle :", person)

    def paint(self, img):
        for person in self.persons:
            person.paint(img)

    def remove_unfollowed_persons(self):
        persons = []
        for person in self.persons:
            if person.is_followed:
                persons.append(person)

        self.persons = persons

    def add_person(self, img, x, y, w, h):
        person = Person(img, x, y, w, h)
        self.persons.append(person)

        # follow in another thread
        FaceFollower(person).start()

        # label in same thread
        self.recognizator.label_new_face(person)
