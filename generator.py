from typing import List, Any

from matplotlib.pylab import *
from matplotlib.animation import FuncAnimation
import sensor_simulator as sen
import aircraft as airc


class Animation:

    def __init__(self, ax, ax2, ax3, ax4, ax5, ax6, air, sensor, t):
        self.ax = ax
        self.ax2 = ax2
        self.ax3 = ax3
        self.ax4 = ax4
        self.ax5 = ax5
        self.ax6 = ax6
        self.air = air
        self.sensor = sensor
        self.t = t
        # set up lines to plot
        # object trace
        self.ln, = self.ax.plot([], [])
        # moving object
        self.ln2, = self.ax.plot([], [], 'ro')
        # velocity
        self.ln3, = self.ax2.plot([], [])
        # acceleration
        self.ln4, = self.ax3.plot([], [])
        # vector products with t(t) & n(t)
        self.ln5, = self.ax4.plot([], [])
        self.ln6, = self.ax5.plot([], [])
        # sensor measurements polar range
        self.ln7, = self.ax6.plot([], [], 'bo', label="range measurement")
        # sensor position
        self.ln8, = self.ax.plot([], [], 'yo', label="sensor position")
        # cartesian
        self.ln9, = self.ax.plot([], [], 'ro', label="cartesian measurement")
        # sensor measurements polar winkel
        self.ln10, = self.ax6.plot([], [], 'ro', label="azimuth")

        # set up data for lines
        self.position = np.zeros((2, self.t))
        self.velocity = np.zeros((1, self.t))
        self.acceleration = np.zeros((1, self.t))
        self.norm = np.zeros((2, 1))
        self.tang = np.zeros((2, 1))
        self.rt = np.zeros((1, self.t))
        self.rn = np.zeros((1, self.t))
        self.time = np.zeros((1, self.t))
        self.cartesian = np.zeros((2, self.t))
        self.polar = np.zeros((2, 1))
        self.polar_vec = np.zeros((2, t))

    def init(self):

        # set up data
        for i in range(self.t):
            self.air.update_stats(i)
            self.position[:, i] = self.air.position.T
            self.velocity[0, i] = np.linalg.norm(self.air.velocity)
            print("air velocity:", self.air.velocity)
            self.acceleration[0, i] = np.linalg.norm(self.air.acceleration)
            print("air acceleration:", self.air.acceleration)
            self.norm = self.air.norm
            self.tang = self.air.tang
            self.rn[0, i] = np.matmul(air.acceleration.T, air.norm)
            self.rt[0, i] = np.matmul(air.acceleration.T, air.tang)
            self.time[0, i] = i
            if i%sensor.delta_t == 0:
                self.sensor.update_stats(i)
                self.cartesian[:, i] = sensor.z_c.T
                z_r = self.sensor.z_p[0]
                print("z_r: \n", z_r)
                print(z_r.shape)
                z_f = self.sensor.z_p[1]
                r = self.sensor.pos
                vec = np.array((np.cos(z_f), np.sin(z_f)))
                vec = vec.reshape((2, 1))
                self.polar = z_r * vec + r
                self.polar_vec[:, i] = self.polar.T

        # set up ax
        self.ax.set_xlim(-12000, 12000)
        self.ax.set_ylim(-12000, 12000)
        self.ax.set_title("aircraft trajectory")
        self.ax.set_xlabel("y in meter")
        self.ax.set_ylabel("x in meter")
        self.ax.grid(True)

        # set up ax2
        self.ax2.set_xlim(-10, 430)
        self.ax2.set_ylim(0, 400)
        self.ax2.set_title("aircraft velocity")
        self.ax2.set_xlabel("r'(t) in meter")
        self.ax2.set_ylabel("t in meter")
        self.ax2.grid(True)

        # set up ax3
        self.ax3.set_xlim(-10, 430)
        self.ax3.set_ylim(-2, 12)
        self.ax3.set_title("aircraft acceleration")
        self.ax3.set_xlabel("r''(t) in m/s²")
        self.ax3.set_ylabel("t in meter")
        self.ax3.grid(True)

        # set up ax4
        self.ax4.set_xlim(-10, 430)
        self.ax4.set_ylim(-10, 10)
        self.ax4.set_title("r''(t)t(t)")
        self.ax4.set_xlabel("r''(t)t(t) in m/s²")
        self.ax4.set_ylabel("t in meter")
        self.ax4.grid(True)

        # set up ax5
        self.ax5.set_xlim(-10, 430)
        self.ax5.set_ylim(-10, 10)
        self.ax5.set_title("r''(t)n(t)")
        self.ax5.set_xlabel("r''(t)n(t) in m/s²")
        self.ax5.set_ylabel("t in meter")
        self.ax5.grid(True)

        #set up ax6
        self.ax6.set_title("sensor polar measurements")
        self.ax6.set_xlabel("time in seconds")
        self.ax6.set_ylabel("range in meter")
        self.ax6.grid(True)
        self.ax6.set_xlim(-24000, 24000)
        self.ax6.set_ylim(-24000, 24000)

        return self.ln, self.ln2, self.ln3, self.ln4, self.ln5, self.ln6, self.ln7, self.ln8, self.ln9, self.ln10

    def update(self, i):
        # update aircraft trajectory
        self.ln.set_data(self.position[0, :i], self.position[1, :i])
        # update aircraft position
        self.ln2.set_data(self.position[0, i], self.position[1, i])

        # update aircraft velocity
        self.ln3.set_data(self.time[0, :i], self.velocity[0, :i])

        # update aircraft acceleration
        self.ln4.set_data(self.time[0, :i], self.acceleration[0, :i])

        # update rt
        self.ln5.set_data(self.time[0, :i], self.rn[0, :i])

        # update rn
        self.ln6.set_data(self.time[0, :i], self.rt[0, :i])

        # update cartesian measurements
        if i%self.sensor.delta_t == 0:
            self.ln9.set_data(self.cartesian[0, :i], self.cartesian[1, :i])

        # plot quiver
        Q1 = ax.quiver(self.position[0], self.position[1], self.tang[0], self.tang[1], units='xy',
                        scale=0.0005, color='g')
        qk1 = ax.quiverkey(Q1, 0.9, 0.9, 2, r'$tangential vector$', labelpos='E',
                           coordinates='figure')
        # plot normal vector
        Q2 = ax.quiver(self.position[0], self.position[1], self.norm[0], self.norm[1], units='xy',
                                 scale=0.0005, color='y')
        qk2 = ax.quiverkey(Q2, 0.9, 0.9, 2, r'$normal vector$', labelpos='E',
                           coordinates='figure')

        # plot polar measurements
        Q3 = ax6.quiver(self.sensor.pos[0], self.sensor.pos[1], self.polar_vec[0, :i], self.polar[1, :i], units='xy',
                           scale=0.1, color='y')
        qk3 = ax6.quiverkey(Q3, 0.9, 0.9, 2, 'polar vector', labelpos='E',
                            coordinates='figure')

        return self.ln, self.ln2, self.ln3, self.ln4, self.ln5, self.ln6, self.ln9, Q1, Q2, Q3


if __name__ == "__main__":

    T = 419
    # set up objects
    air = airc.Aircraft(300, 9, 0)
    sensor_position = np.array((0, 0))
    sensor_position = sensor_position.reshape((2, 1))
    # o.2° is 0.00349066 radiant
    sensor = sen.Sensor(50, 20, 0.00349066, 5, sensor_position, air)

    # set up figure and subplots
    font = {'size': 9}
    matplotlib.rc('font', **font)
    fig = figure(num=0, figsize=(16, 9))  # , dpi = 100)
    fig.suptitle("ground truth generator", fontsize=12)
    ax = plt.subplot2grid((3, 3), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
    ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
    ax4 = plt.subplot2grid((3, 3), (2, 0))
    ax5 = plt.subplot2grid((3, 3), (2, 1))
    ax6 = plt.subplot2grid((3, 3), (0, 2))

    # plot animation
    ani = Animation(ax, ax2, ax3, ax4, ax5, ax6, air, sensor, T)
    animation = FuncAnimation(fig, ani.update, frames=T,
                              init_func=ani.init, interval=25,  blit=True, repeat=True)

    ax.legend()
    ax6.legend()
    plt.tight_layout()
    plt.show()
