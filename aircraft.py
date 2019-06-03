import numpy as np


class Aircraft:

    def __init__(self, v, q, t):
        self.v = v
        self.q = q
        self.t = t
        self.position = np.zeros(2)
        self.velocity = np.zeros(2)
        self.acceleration = np.zeros(2)
        self.tang = np.zeros(2)
        self.norm = np.zeros(2)

    def get_w_a(self, t):
        """Returns matrices w and a"""
        a = (self.v**2 / self.q)
        w = self.q / (2 * self.v)
        return w, a

    def get_position(self, t):
        """Returns the current position of aircraft for time instant t"""
        w, a = self.get_w_a(t)
        dir = np.array([np.sin(w * t), np.sin(2 * w * t)])
        dir = a * dir
        return dir

    def get_velocity(self, t):
        """Returns the current velocity of aircraft for time instant t"""
        w, a = self.get_w_a(t)
        dir = np.array([np.cos(w * t)/2, np.cos(2 * w * t)])
        dir = self.v * dir
        return dir

    def get_acceleration(self, t):
        """Returns the current acceleration of aircraft for time instant t"""
        w, a = self.get_w_a(t)
        dir = np.array([np.sin(w * t) / 4, np.sin(2 * w * t)])
        dir = -self.q * dir
        return dir

    def get_tangential(self, t):
        """Returns the current tangential vector of the aircraft trajectory for time instant t"""
        v = self.get_velocity(t)
        tang = (1 / np.linalg.norm(v)) * v
        return tang

    def get_normal(self, t):
        """Returns the current normal vector of the aircraft trajectory for time instant t"""
        v = self.get_velocity(t)
        norm = (1/np.linalg.norm(v)) * np.array([-v[1], v[0]])
        return norm

    def update_stats(self, t):
        """Updates all aircraft values for time instant t"""
        self.t = t
        self.position = self.get_position(t)
        self.velocity = self.get_velocity(t)
        self.acceleration = self.get_acceleration(t)
        self.tang = self.get_tangential(t)
        self.norm = self.get_normal(t)