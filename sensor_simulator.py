import numpy as np
import generator as gen
import types


class Sensor:

    def __init__(self, sigma, delta_t, x, air):
        self.sigma = sigma
        self.delta_t = delta_t
        self.x = x
        self.air = air
        self.z_c = np.zeros((2, 3))
        self.z_p = np.zeros((2, 3))

    def update_x(self, t):
        self.air.update_stats(t)
        self.x = np.transpose(self.air.position)
        self.x = np.hstack((self.x, np.transpose(self.air.velocity)))
        self.x = np.hstack((self.x, np.transpose(self.air.acceleration)))
        self.x = np.transpose(self.x)
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
        self.x = self.update_x(t)
        prod = np.matmul(H, self.x)
        u = self.get_u(t)
        z = prod + 1000*u

        # test prints
        print("x:\n", self.x)
        print("H:\n", H)
        print("product H*x:\n", prod)
        print("product u; sigma * normrnd(0,1):\n", u)
        print("sum: H*x + sigma*normrnd(0,1):\n", z)

        return z

    def range(self, t):
        pass


def main():

    aircraft = gen.Aircraft(300, 9, 0)
    x = np.zeros((2, 3))
    sensor = Sensor(50, 5, x, aircraft)
    sensor.update_x(42)
    sensor.cartesian(42)

    # test to use sensor simulator in truth ground generator
    # overwrite functions of object air
    #aircraft.get_position = types.MethodType(sensor.cartesian, aircraft)
    #print(aircraft.get_position())
    #gen.main(aircraft)


if __name__ == "__main__":
    main()