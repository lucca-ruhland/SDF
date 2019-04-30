import numpy as np
import generator as gen


class Sensor:

    def __init__(self, sigma_c, sigma_r, sigma_f, delta_t, pos, air):
        # constants
        self.sigma_c = sigma_c
        self.sigma_r = sigma_r
        self.sigma_f = sigma_f
        self.delta_t = delta_t
        self.pos = pos
        # objects
        self.air = air
        # variables
        self.x = np.zeros((6, 1))
        self.z_c = np.zeros((2, 3))
        self.z_p = np.zeros((2, 3))


    def update_x(self, t):
        self.air.update_stats(t)
        self.x = np.transpose(self.air.position)
        self.x = np.hstack((self.x, np.transpose(self.air.velocity)))
        self.x = np.hstack((self.x, np.transpose(self.air.acceleration)))
        self.x = np.transpose(self.x)
        return self.x



    def get_u(self, sg1, sg2, t):
        u = np.zeros((2, 1))
        u[0] = sg1 * np.random.normal(0, 1)
        u[1] = sg2 * np.random.normal(0, 1)
        return u

    def cartesian(self, t):
        z = np.zeros((2, 1))
        H = np.zeros((2, 6))
        H[0:2, 0:2] = np.eye(2, dtype=int)
        self.x = self.update_x(t)
        prod = np.matmul(H, self.x)
        u = self.get_u(self.sigma_c, self.sigma_c, t)
        z = prod + u

        # test prints
        print("x:\n", self.x)
        print("H:\n", H)
        print("product H*x:\n", prod)
        print("product u; sigma * normrnd(0,1):\n", u)
        print("sum: H*x + sigma*normrnd(0,1):\n", z)

        return z

    def range(self, t):
        z = np.zeros((2, 1))
        self.x = self.update_x(t)
        z[0] = np.sqrt((self.x[0] - self.pos[0])**2 + (self.x[1] - self.pos[1])**2)
        z[1] = np.arctan((self.x[1] - self.pos[1]) / (self.x[0] - self.pos[0]))
        u = self.get_u(self.sigma_r, self.sigma_f, t)
        print("z before:\n", z)
        print("u:\n", u)
        z = z + u
        print("z + u:\n", z)
        return z

    def update_stats(self, t):
        self.z_c = self.cartesian(t)
        self.z_p = self.range(t)
        self.x = self.update_x(t)


def main():

    aircraft = gen.Aircraft(300, 9, 0)
    # defining sensor position
    sensor_pos = np.zeros((2, 1))
    sensor_pos[0] = 0
    sensor_pos[1] = 0
    # create sensor object for test
    sensor = Sensor(50, 20, 0.2, 5, sensor_pos, aircraft)
    # test
    sensor.update_x(42)
    sensor.cartesian(42)
    sensor.range(42)

if __name__ == "__main__":
    main()