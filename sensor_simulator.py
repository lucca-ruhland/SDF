import numpy as np
import ground_truth_generator as gen

class Sensor:
    def __init__(self, sigma, delta_t, x, air):
        self.sigma = sigma
        self.delta_t = delta_t
        self.x = x
        self.air = air
        self.z_c = np.zeros((2, 3))
        self.z_p = np.zeros((2, 3))

    def update_x(self, t):
        self.x = np.transpose(self.air.get_position(t))
        self.x = np.hstack((self.x, np.transpose(self.air.get_velocity(t))))
        self.x = np.hstack((self.x, np.transpose(self.air.get_acceleration(t))))
        self.x = np.transpose(self.x)
        print("x:")
        print(self.x)
        return self.x

    def get_u(self, t):
        u = np.zeros((2, 1))
        u[0] = np.random.normal(0, 1)
        u[1] = np.random.normal(0, 1)
        u = self.sigma * u
        return u

    def sensor_position(self, t):
        r = self.air.get_position(t)
        return r

    def cartesian(self, t):
        z = np.zeros((2, 1))
        H = np.zeros((2, 6))
        H[0:2, 0:2] = np.eye(2, dtype=int)
        print(H)
        self.x = self.update_x(t)
        prod = np.matmul(H, self.x)
        print(prod)
        u = self.get_u(t)
        z = prod + u
        print("u:")
        print(u)
        print("z:")
        print(z)
        return z

    def range(self, t):
        pass



def main():
    aircraft = gen.Aircraft(300, 9, 0)
    x = np.zeros((2, 3))
    sensor = Sensor(50, 5, x, aircraft)
    sensor.update_x(42)
    sensor.cartesian(42)

if __name__ == "__main__":
    main()