import numpy as np
from aircraft import Aircraft


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
        # simulated measurements
        self.z_c = np.zeros(2)
        self.z_r = np.zeros(2)
        self.z_az = np.zeros(2)
        # target components
        self.x_pos = np.zeros(2)
        self.x_v = np.zeros(2)
        self.x_q = np.zeros(2)

    def update_x(self, t):
        """Updates for given time instant t all aircraft components"""
        self.air.update_stats(t)
        self.x_pos = self.air.position
        self.x_v = self.air.velocity
        self.x_q = self.air.acceleration

    def get_u(self, sg1, sg2):
        """returns Matrix for normal distribution"""
        u = np.array([sg1 * np.random.normal(0, 1), sg2 * np.random.normal(0, 1)])
        return 10 * u  # multiplier making sensor measurements more inaccurate

    def cartesian(self, t):
        """Returns the cartesian measurements of the aircraft for time instanct t"""
        self.update_x(t)
        u = self.get_u(self.sigma_c, self.sigma_c)
        z = self.x_pos + u
        # test prints
        # print("product u; sigma * normrnd(0,1):\n", u)
        # print("sum: H*x + sigma*normrnd(0,1):\n", z)
        return z

    def range(self, t):
        """returns the range between sensor and aircraft for time instant t"""
        self.update_x(t)
        u = self.get_u(self.sigma_r, self.sigma_f)[0]
        z = np.array([np.sqrt((self.x_pos[0] - self.pos[0])**2 + (self.x_pos[1] - self.pos[1])**2)])
        z = z + u
        return z

    def azimuth(self, t):
        """returns the azimuth of the aircraft towards the sensor for time instant t"""
        self.update_x(t)
        u = self.get_u(self.sigma_r, self.sigma_f)[1]
        z = np.array([np.arctan2((self.x_pos[1] - self.pos[1]), (self.x_pos[0] - self.pos[0]))])
        z = z + u
        return z

    def update_stats(self, t):
        """Updates all measurements of sensor for given time instant t"""
        self.update_x(t)
        self.z_c = self.cartesian(t)
        self.z_r = self.range(t)
        self.z_az = self.azimuth(t)


def main():

    aircraft = Aircraft(300, 9, 0)
    # defining sensor position
    sensor_pos = np.zeros((2, 1))
    sensor_pos[0] = 0
    sensor_pos[1] = 0
    # create sensor object for test
    sensor = Sensor(50, 20, 0.2, 5, sensor_pos, aircraft)

    for i in range(10):
        sensor.update_stats(i)
        obj_pos = np.array([sensor.z_c]).reshape((2, 1))
        r = sensor.z_r
        az = sensor.z_az
        z = r * np.array([np.cos(az), np.sin(az)]).reshape((2, 1)) + sensor_pos
        print("object position: \n", obj_pos)
        print("recalculated position:\n", z)
        dif = obj_pos - z
        print("Differences:\n", dif)

if __name__ == "__main__":
    main()