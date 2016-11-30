import math
import numpy as np
import cv2


class Position:
    """A simple example class"""

    def __init__(self, x, y, w, h):
        self.measures = []
        self.measures.append((x, y))
        self.predictions = []

    def corresponds(self, x, y, epsilon):
        corresponds = False
        diff_meas_x  = math.fabs(self.measures[len(self.measures) - 1][0] - x)
        diff_meas_y = math.fabs(self.measures[len(self.measures) - 1][1] - y)

        diff_pred_x  = math.fabs(self.predictions[len(self.predictions) - 1][0] - x)
        diff_pred_y = math.fabs(self.predictions[len(self.predictions) - 1][1] - y)

        if diff_meas_x < epsilon and diff_meas_y < epsilon:
            corresponds = True
        elif len(self.predictions) > 0 and diff_pred_x < epsilon and diff_pred_y < epsilon:
            corresponds = True

        return corresponds


    def corresponds(self, x, y, w, h, epsilon):
        corresponds = False
        diff_meas_x  = math.fabs(self.measures[len(self.measures) - 1][0] - (x + w/2))
        diff_meas_y = math.fabs(self.measures[len(self.measures) - 1][1] - (y + h/2))

        diff_pred_x  = math.fabs(self.predictions[len(self.predictions) - 1][0] - x)
        diff_pred_y = math.fabs(self.predictions[len(self.predictions) - 1][1] - y)

        if diff_meas_x < epsilon and diff_meas_y < epsilon:
            corresponds = True
        elif len(self.predictions) > 0 and diff_pred_x < epsilon and diff_pred_y < epsilon:
            corresponds = True

        return corresponds

    def clear_old_measures(self):
        if (len(self.measures) > 100):
            del self.measures[:25]

    def add_measure(self, x, y):
        self.measures.append((int(x), int(y)))

    def add_prediction(self, x, y):
        self.predictions.append((x, y))

        if (len(self.predictions) > 100):
            del self.predictions[:25]

    def clear_old_predictions(self):
        if (len(self.predictions) > 100):
            del self.predictions[:25]

    def reset(self):
        print("do nothing")

    def run_predict(self):
        kalman = cv2.KalmanFilter(4, 2)
        kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 0.03

        while True:
            kalman.correct(self.mp)
            tp = kalman.predict()
            self.predictions.append((int(tp[0]), int(tp[1])))

            k = cv2.waitKey(30) & 0xFF

            if k == 27: break
            if k == 32: self.reset()
