import numpy as np
from sensor_simulator import Sensor
from aircraft import Aircraft
import numpy as np

class kalman:
    def __init__(self, aircraft, sigma, *argv):
        """Initialize local variables and measurement data for filter"""
        # initialize objects
        self.air = aircraft
        # self.sen = sensor

        # self.delta_t = sensor.delta_t
        # self.sigma_c = sigma_c
        self.delta_t = 5
        self.sigma = sigma
        self.sensors = argv

        # init local vars
        self.x = np.zeros((4, 1))
        # self.R = self.sigma * np.eye(2)
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
        self.D = self.sigma**2 * self.D
        print("D:\n", self.D)

        self.H = np.array([1, 0, 0, 0,
                           0, 1, 0, 0])
        self.H = self.H.reshape(2, 4)
        self.S = np.zeros((2, 2))

        self.R = np.zeros((2, 2))
        for arg in argv:
            if isinstance(arg, Sensor):
                self.R = self.R + np.linalg.inv(arg.sigma_c**2 * np.eye(2))
        self.R = np.linalg.inv(self.R)
        print("R^-1:", self.R)


    def prediction(self, t):
        """Returns a prediction based on a dynamic model of sensor measurements"""
        self.air.update_stats(t)
        self.x = np.array([self.z[0], self.z[1], self.air.velocity[0], self.air.velocity[1]]).reshape(4, 1)
        self.x = np.dot(self.F, self.x)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.D

    def filtering(self, t):
        """Returns a filtered dynamic model"""
        self.fusion(t)
        self.v = self.z - np.dot(self.H, self.x)

        self.S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        self.W = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(self.S))
        self.x = self.x + np.dot(self.W, self.v)
        self.P = self.P - np.dot(np.dot(self.W, self.S), self.W.T)

    def fusion(self, t):
        self.z = np.zeros((2, 1))
        for arg in self.sensors:
            if isinstance(arg, Sensor):
                _r = arg.sigma_c**2 * np.eye(2)
                arg.update_stats(t)
                _z = np.array([arg.z_c[0], arg.z_c[1]]).reshape(2, 1)
                self.z = self.z + np.dot(np.linalg.inv(_r), _z)
        self.z = np.matmul(self.R, self.z)
        print("fusion:\n", self.z)


    def calc(self, t):
        self.prediction(t)
        self.fusion(t)
        self.filtering(t)


if __name__ == "__main__":
    # run tests for kalman filter
    # init objects needed for kalman filter
    air = Aircraft(300, 9, 0)
    sensor_position = np.array((-8000, 4000)).reshape((2, 1))
    sensor1 = Sensor(50, 20, 0.00349066, 5, sensor_position, air)
    sensor_position2 = np.array((0, 0)).reshape((2, 1))
    sensor2 = Sensor(50, 20, 0.00349066, 5, sensor_position2, air)
    kalman_filter = kalman(air, 50, 5, sensor1, sensor2)

    kalman_filter.fusion(200)
    print(kalman_filter.z)
    for i in range(1, 100):
        kalman_filter.calc(i)
        print(kalman_filter.x)



