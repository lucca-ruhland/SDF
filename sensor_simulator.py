import numpy as np
import ground_truth_generator as gen

class Sensor:
    def __init__(self, sigma, delta_t, x):
        self.sigma = sigma
        self.delta_t = delta_t
        self.x = x
        self.z_c = np.zeros((2, 3))
        self.z_p = np.zeros((2, 3))

    def update_x(self, t):
        air = gen.Aircraft(300, 9, 0)
        self.x[0:2, 0:1] = air.get_position(t)
        self.x[0:2, 1:2] = air.get_velocity(t)
        self.x[0:2, 2:3] = air.get_acceleration(t)
        return self.x

    def cartesian(self, t):
        H = np.zeros((2, 6))
        H[0:2, 0:2] = np.eye(2, dtype=int)
        print(H)
        self.x = self.update_x(t)
        print(self.x)
        x_t = np.transpose(self.x)
        print(x_t)
        prod = np.matmul(H, x_t)
        print(prod)

def main():
    x = np.zeros((2, 3))
    sensor = Sensor(50, 5, x)
    sensor.update_x(42)
    sensor.cartesian(42)

if __name__ == "__main__":
    main()