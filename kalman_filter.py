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
        # state vector
        # self.x = np.zeros((4, 1))
        self.x = self.init_state_vecor(0)
        # self.R = self.sigma * np.eye(2)
        self.z = np.zeros((2, 1))
        # Covariance P of pdf
        self.P = np.zeros((4, 4))

    def get_covariance_r(self):
        R = np.zeros((2, 2))
        for arg in self.sensors:
            if isinstance(arg, Sensor):
                R = R + np.linalg.inv(arg.sigma_c ** 2 * np.eye(2))
        return np.linalg.inv(R)

    def get_h(self):
        return np.array([1, 0, 0, 0, 0, 1, 0, 0]).reshape(2, 4)

    def get_covariance_d(self):
        D = np.array([0.25 * self.delta_t ** 4, 0, 0.5 * self.delta_t ** 3, 0,
                      0, 0.25 * self.delta_t ** 4, 0, 0.5 * self.delta_t ** 3,
                      0.5 * self.delta_t ** 3, 0, self.delta_t ** 2, 0,
                      0, 0.5 * self.delta_t ** 3, 0, self.delta_t ** 2]).reshape((4, 4))
        return self.sigma ** 2 * D

    def get_dynamics_f(self):
        return np.array([1, 0, self.delta_t, 0, 0, 1, 0, self.delta_t, 0, 0, 1, 0, 0, 0, 0, 1]).reshape((4, 4))

    def init_state_vecor(self, t):
        self.air.update_stats(t)
        return np.array([self.air.position[0], self.air.position[1], self.air.velocity[0], self.air.velocity[1]]).reshape((4, 1))

    def prediction(self, t):
        """Returns a prediction based on a dynamic model of sensor measurements"""
        self.air.update_stats(t)
        x = self.x
        F = self.get_dynamics_f()
        D = self.get_covariance_d()
        x = np.dot(F, x)
        P = np.dot(np.dot(F, self.P), F.T) + D
        return x, P

    def filtering(self, t):
        """Returns a filtered dynamic model"""
        self.fusion(t)
        H = self.get_h()
        R = self.get_covariance_r()
        v = self.z - np.dot(H, self.x)

        S = np.dot(np.dot(H, self.P), H.T) + R
        W = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))
        x = self.x + np.dot(W, v)
        P = self.P - np.dot(np.dot(W, S), W.T)
        return x, P

    def fusion(self, t):
        z = np.zeros((2, 1))
        R = self.get_covariance_r()
        for arg in self.sensors:
            if isinstance(arg, Sensor):
                _r = arg.sigma_c**2 * np.eye(2)
                arg.update_stats(t)
                _z = np.array([arg.z_c[0], arg.z_c[1]]).reshape(2, 1)
                z = z + np.dot(np.linalg.inv(_r), _z)
        z = np.matmul(R, z)
        print("fusion:\n", z)
        return z


    def update_cartesian(self, t):
        self.x, self.P = self.prediction(t)
        self.z = self.fusion(t)
        self.x, self.P = self.filtering(t)


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
        kalman_filter.update_cartesian(i)
        print(kalman_filter.x)



