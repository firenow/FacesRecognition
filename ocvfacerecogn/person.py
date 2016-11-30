#!/usr/bin/env python

import uuid
import math
import cv2
import time
import random
import numpy as np
from descriptor import Descriptor
from position import Position


class Person:
    """A simple Person class"""

    def __init__(self, img, x, y, w, h):
        self.id = uuid.uuid1()
        self.descriptor = Descriptor(img, x, y, w, h)
        self.position = Position(x, y, w, h)
        self.last_seen = time.time()
        self.color = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))
        self.is_followed = False

    def add_position(self, x, y):
        self.position.add_measure(x, y)
        # self.position.clear_old_measures()
        self.last_seen = time.time()
        print("%s suivie à %s" % (self, self.last_seen))

    def add_position(self, x, y, w, h):
        self.position.add_measure(x + w/2, y + h/2)
        # self.position.clear_old_measures()
        self.last_seen = time.time()
        formatted_now = time.ctime(int(self.last_seen))
        print("%s suivie à %s" % (self, str(formatted_now)))

    def add_prediction(self, x, y):
        self.position.add_prediction(x, y)
        # self.position.clear_old_predictions()

    def get_face(self):
        return self.descriptor.get_image()

    def is_identified(self):
        return self.descriptor.is_identified()

    def set_prediction_result(self, result):
        print("\nPersonne identifiée : {0} avec une confiance de {1}% \n".format(result.getLabel(), result.getDist()))
        self.descriptor.set_label(result.getLabel(), result.getDist())

    def corresponds(self, x, y, epsilon):
        return self.position.corresponds(x, y, epsilon)

    def corresponds(self, x, y, w, h, epsilon):
        return self.position.corresponds(x, y, w, h, epsilon)

    def paint(self, image):
        for i in range(len(self.position.measures) - 1):
            cv2.line(image, self.position.measures[i], self.position.measures[i + 1], self.color)

    def get_last_measurement(self):
        return np.array([np.float32(self.position.measures[len(self.position.measures) - 1][0]), np.float32(self.position.measures[len(self.position.measures) - 1][1])])

    def __str__(self):
        return "Personne " + self.descriptor.name
