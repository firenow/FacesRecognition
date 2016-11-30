import cv2
import math


class Descriptor:
    """A simple example class"""
    def __init__(self, img, x, y, w, h):
        self.img = img
        self.img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.x0 = x
        self.y0 = y
        self.w0 = w
        self.h0 = h

        self.name = "UKN"
        self.confidence = 1

    def is_identified(self):
        return self.name != "UKN"

    def set_label(self, name, confidence):
        self.name = str(name)
        self.confidence = confidence

    def get_image(self):
        return self.img_gray[self.y0:self.y0+self.h0,self.x0:self.x0+self.w0]