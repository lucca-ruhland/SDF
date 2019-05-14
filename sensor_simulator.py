import numpy as np
import aircraft as airc


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
        self.x = np.zeros(6)
        self.z_c = np.zeros(2)
        self.z_p = np.zeros(2)

    def update_x(self, t):
        self.air.update_stats(t)
        # print("position:\n", self.air.position)
        # print(self.air.position.shape)
        # print("velocity:\n", self.air.velocity)
        # print(self.air.velocity.shape)
        # print("acceleration:\n", self.air.acceleration)
        # print(self.air.acceleration.shape)
        self.x = np.transpose(self.air.position)
        self.x = np.hstack((self.x, np.transpose(self.air.velocity)))
        self.x = np.hstack((self.x, np.transpose(self.air.acceleration)))
        self.x = np.transpose(self.x)
        # print("x:\n", self.x)
        # print(self.x.shape)
        return self.x

    def get_u(self, sg1, sg2):
        u = np.array([sg1 * np.random.normal(0, 1), sg2 * np.random.normal(0, 1)])
        return u

    def cartesian(self, t):
        self.x = self.update_x(t)
        u = self.get_u(self.sigma_c, self.sigma_c)
        pos = self.x[0:2]
        z = pos + u

        # test prints
        # print("product u; sigma * normrnd(0,1):\n", u)
        # print("sum: H*x + sigma*normrnd(0,1):\n", z)

        return z

    def range(self, t):
        z = np.zeros((2, 1))
        self.x = self.update_x(t)
        zx = np.sqrt((self.x[0:1] - self.pos[0])**2 + (self.x[1:2] - self.pos[1])**2)
        zy = np.arctan((self.x[1:2] - self.pos[1]) / (self.x[0:1] - self.pos[0]))
        u = self.get_u(self.sigma_r, self.sigma_f)
        # print("z before:\n", z)
        # print("u:\n", u)
        z = np.array([zx, zy])
        z = z + u
        # print("z + u:\n", z)
        return z

    def update_stats(self, t):
        self.z_c = self.cartesian(t)
        self.z_p = self.range(t)
        self.x = self.update_x(t)


def main():

    aircraft = airc.Aircraft(300, 9, 0)
    # defining sensor position
    sensor_pos = np.zeros((2, 1))
    sensor_pos[0] = 0
    sensor_pos[1] = 0
    # create sensor object for test
    sensor = Sensor(50, 20, 0.2, 5, sensor_pos, aircraft)
    # test
    sensor.update_x(42)
    sensor.cartesian(42)
    for i in range(10):
        sensor.cartesian(i)
    sensor.range(42)

if __name__ == "__main__":
    main()