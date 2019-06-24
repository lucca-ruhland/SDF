from sensor_simulator import Sensor
import numpy as np
from numpy.linalg import inv


class KalmanFilter:

    def __init__(self, aircraft, sigma, sensor_list):
        """Initialize local variables and measurement data for filter"""
        # initialize objects
        self.air = aircraft
        # init sensors
        self.delta_t = 5
        self.sigma = sigma
        # save all sensors
        self.sensors = sensor_list

        # init local vars
        # state vector
        # self.x = np.zeros((4, 1))
        self.x = self.init_state_vecor(0)
        self.z = np.zeros((2, 1))
        self.z_polar = np.zeros((2, 1))
        # Covariance P of pdf
        self.P = np.zeros((4, 4))
        # covariance R
        self.R = np.zeros((2, 2))

        # list of all state vectors x for retrodiction
        # list_x = [x0|0,   x1|0,   x1|1, ... ]
        self.list_x = []
        self.list_x.append(self.x)
        # list of all covariances P for retrodiction
        # list_p = [p0|0,   p1|0,   p1|1 ...]
        self.list_p = []
        self.list_p.append(self.P)

    def get_covariance_r(self):
        """Harmonic mean of measurement covariance"""
        """Returns numpy array of dimension 2x2"""
        R = np.zeros((2, 2))
        for arg in self.sensors:
            if isinstance(arg, Sensor):
                R = R + inv(arg.sigma_c ** 2 * np.eye(2))
        return inv(R)

    def get_covariance_r_polar(self):
        """Harmonic mean of measurement covariance"""
        """Returns numpy array of dimension 2x2"""
        R = np.zeros((2, 2))
        for arg in self.sensors:
            if isinstance(arg, Sensor):
                R = R + inv(np.diag((arg.sigma_r**2, arg.sigma_f**2)))
        return inv(R)

    def get_covariance_polar_to_cartesian(self, z_polar):
        """Harmonic mean of measurement covariance"""
        """Transforms polar measurements to cartesian"""
        """Returns numpy array of dimension 2x2"""
        r = z_polar[0]
        f = z_polar[1]
        D = np.array([np.cos(f), -np.sin(f), np.sin(f), np.cos(f)]).reshape((2, 2))
        S = np.diag((1, r))
        T = np.dot(D, S)
        R = self.get_covariance_r_polar()
        return np.dot(np.dot(T, R), T.T)

    def get_h(self):
        """Returns matrix to transform state vector x to measurement dimension"""
        """Returns numpy array of dimension 2x4"""
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
        """Returns an initialized state vector for beginning e.g. x0|0"""
        """Returns numpy array of dimension 4x1"""
        self.air.update_stats(t)
        return np.array([self.air.position[0], self.air.position[1], self.air.velocity[0], self.air.velocity[1]]).reshape((4, 1))

    def prediction(self, t):
        """Returns a prediction based on a dynamic model of sensor measurements"""
        """Returns two numpy arrays of dimension 4x1 (matrix x) and 4x4 (matrix P)"""
        self.air.update_stats(t)
        x = self.x
        F = self.get_dynamics_f()
        D = self.get_covariance_d()
        x = np.dot(F, x)
        P = np.dot(np.dot(F, self.P), F.T) + D
        return x, P

    def filtering(self, t):
        """Returns a filtered dynamic model"""
        """Returns two numpy arrays of dimension 4x1 (matrix x) and 4x4 (matrix P)"""
        H = self.get_h()
        R = self.get_covariance_r()
        v = self.z - np.dot(H, self.x)

        S = np.dot(np.dot(H, self.P), H.T) + R
        S = S.astype(float)  # to fix inv(S) for polar measurements
        W = np.dot(np.dot(self.P, H.T), inv(S))
        x = self.x + np.dot(W, v)
        P = self.P - np.dot(np.dot(W, S), W.T)
        return x, P

    def fusion(self, t):
        """Combine measurements of all sensors into a single measurement"""
        """Returns a numpy array with dimension 2x1"""
        z = np.zeros((2, 1))
        R = self.get_covariance_r()
        for arg in self.sensors:
            if isinstance(arg, Sensor):
                arg.update_stats(t)
                _r = arg.sigma_c**2 * np.eye(2)
                _z = np.array([arg.z_c[0], arg.z_c[1]]).reshape(2, 1)
                z = z + np.dot(inv(_r), _z)
        z = np.matmul(R, z)
        return z

    def fusion_polar(self, t):
        """Combine measurements of all sensors into a single measurement"""
        """Used for measurements of sensors using range and azimuth"""
        """Returns two numpy array with dimension 2x1 (z) and 2x2 (covariance R)"""
        z = np.zeros((2, 1))
        R = np.zeros((2, 2))
        for arg in self.sensors:
            if isinstance(arg, Sensor):
                arg.update_stats(t)
                # z in polar to calculate covariance r
                _z = np.array([arg.z_r, arg.z_az]).reshape((2, 1))
                _r = self.get_covariance_polar_to_cartesian(_z)
                # transform z into cartesian and add sensor position to get total and not relative coordinates
                _z = arg.z_r * np.array([np.cos(arg.z_az), np.sin(arg.z_az)]).reshape((2, 1)) + arg.pos
                _r = _r.astype(float)  # to make inversion possible
                R = R + inv(_r)
                z = z + np.dot(inv(_r), _z)
        R = inv(R)
        z = np.matmul(R, z)
        return z, R

    def update_cartesian(self, t):
        """Calculate Kalman Filter based on cartesian sensor measurements for time instance t"""
        """Updates class variables x(prediction), P(covariance) and z(measurement)"""
        self.x, self.P = self.prediction(t)
        self.z = self.fusion(t)
        self.x, self.P = self.filtering(t)

    def update_polar(self, t):
        """Calculate Kalman Filter based on polar sensor measurements for time instance t"""
        """Updates class variables x(prediction), P(covariance), z(measurement) and R(covariance)"""
        self.x, self.P = self.prediction(t)
        self.z, self.R = self.fusion_polar(t)
        self.x, self.P = self.filtering(t)