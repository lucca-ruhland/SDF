import numpy as np
from sensor_simulator import Sensor
from aircraft import Aircraft
import numpy as np

class kalman:
    def __init__(self, aircraft, sensor, sigma_c, sigma, *argv):
        """Initialize local variables and measurement data for filter"""
        # initialize objects
        self.air = aircraft
        self.sen = sensor

        self.delta_t = sensor.delta_t
        self.sigma_c = sigma_c
        self.sigma = sigma

        # init local vars
        self.x = np.zeros((4, 1))
        self.R = self.sigma * np.eye(2)
        self.z = np.zeros((2, 1))
        self.v = np.zeros((2, 1))
        self.W = np.zeros((4, 2))
        # Covariance P of pdf
        self.P = np.zeros((4, 4))
        self.F = np.array([1, 0, self.delta_t, 0, 0, 1, 0, self.delta_t, 0, 0, 1, 0, 0, 0, 0, 1])
        self.F.shape = (4, 4)

        self.D = np.array([0.25 * self.delta_t**4, 0, 0.5 * self.delta_t**3, 0,
                           0, 0.25*self.delta_t**4, 0, 0.5 * self.delta_t**3,
                           0.5 * self.delta_t**3, 0, self.delta_t**2, 0,
                           0, 0.5 * self.delta_t**3, 0, self.delta_t**2])
        self.D.shape = (4, 4)

        self.H = np.array([1, 0, 0, 0,
                           0, 1, 0, 0])
        self.H = self.H.reshape(2, 4)
        self.S = np.zeros((2, 2))


    def prediction(self, t):
        """Returns a prediction based on a dynamic model of sensor measurements"""
        self.sen.update_stats(t-1)
        self.x = np.array([self.sen.z_c[0], self.sen.z_c[1], self.sen.x_v[0], self.sen.x_v[1]]).reshape(4, 1)
        self.x = np.matmul(self.F, self.x)
        self.P = np.matmul(np.matmul(self.F, self.P), self.F.T) + self.D

    def filtering(self, t):
        """Returns a filtered dynamic model"""
        self.sen.update_stats(t)
        self.z = self.sen.z_c.reshape(2, 1)
        self.v = self.z - np.matmul(self.H, self.x)

        self.S = np.matmul(np.matmul(self.H, self.P), self.H.T) + self.R
        self.W = np.matmul(np.matmul(self.P, self.H.T), np.linalg.inv(self.S))
        self.x = self.x + np.matmul(self.W, self.v)
        self.P = self.P - np.matmul(np.matmul(self.W, self.S), self.W.T)

    def calc(self, t):
        self.prediction(t)
        self.filtering(t)


if __name__ == "__main__":
    # run tests for kalman filter
    # init objects needed for kalman filter
    air = Aircraft(300, 9, 0)
    sensor_position = np.array((-5000, 0))
    sensor_position = sensor_position.reshape((2, 1))
    sensor = Sensor(50, 20, 0.00349066, 5, sensor_position, air)
    kalman_filter = kalman(air, sensor, sensor.sigma_c, 5)


    for i in range(1, 421, 5):
        kalman_filter.calc(i)
        print("i:\n", i)
        print("x:", kalman_filter.x)



