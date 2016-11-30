import cv2, numpy as np
from person import Person
import threading
import time


class FaceFollower(threading.Thread):
    '''
    Follow a given person accross the application
    '''

    def __init__(self, person):
        super(FaceFollower, self).__init__()
        self.person = person
        self.timeout = 5

    def should_a_prediction_occurs(self):
        is_ok_timeout = (time.time() - self.person.last_seen) < self.timeout
        return is_ok_timeout

    def run(self):
        print("\nSuivi de la personne ...\n")
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03

        mp = self.person.get_last_measurement()

        while self.should_a_prediction_occurs():
            self.person.is_followed = True
            kalman.correct(mp)
            tp = kalman.predict()
            self.person.add_prediction(int(tp[0]), int(tp[1]))

            k = cv2.waitKey(30) & 0xFF

            if k == 27: break

        formatted_now = time.ctime(int(self.person.last_seen))

        print("\nPerdu %s de vue à %s : arrêt du suivi\n" % (self.person, str(formatted_now)))
        self.person.is_followed = False;