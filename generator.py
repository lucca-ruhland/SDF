from typing import List, Any

from matplotlib.pylab import *
from matplotlib.animation import FuncAnimation
import sensor_simulator as sen
#1matplotlib.use('GTKAgg')


class Aircraft:

    def __init__(self, v, q, t):
        self.v = v
        self.q = q
        self.t = t
        self.position = np.zeros((2, 1))
        self.velocity = np.zeros((2, 1))
        self.acceleration = np.zeros((2, 1))
        self.tang = np.zeros((2, 1))
        self.norm = np.zeros((2, 1))

    def get_w_a(self, t):
        A = (self.v ** 2) / self.q
        w = self.q / (2 * self.v)
        return w, A

    def get_position(self, t):
        r_t = np.zeros((2, 1))
        w, A = self.get_w_a(t)
        x_t = A * np.sin(w * t)
        y_t = A * np.sin(2 * w * t)
        r_t[0] = x_t
        r_t[1] = y_t
        return r_t

    def get_velocity(self, t):
        r_i = np.zeros((2, 1))
        w, A = self.get_w_a(t)
        r_i[0] = self.v * np.cos(w * t) / 2
        r_i[1] = self.v * np.cos(2 * w * t)
        return r_i

    def get_acceleration(self, t):
        r_ii = np.zeros((2, 1))
        w, A = self.get_w_a(t)
        r_ii[0] = -self.q * np.sin(w * t) / 4
        r_ii[1] = self.q * np.sin(2 * w * t)
        return r_ii

    def get_tangential(self, t):
        tang = np.zeros((2, 1))
        r_i = self.get_velocity(t)
        tang[0] = (1/(np.linalg.norm(r_i))) * r_i[0]
        tang[1] = (1/(np.linalg.norm(r_i))) * r_i[1]
        return tang

    def get_normal(self, t):
        n = np.zeros((2, 1))
        r_i = self.get_velocity(t)
        n[0] = (1 / (np.linalg.norm(r_i))) * -r_i[1]
        n[1] = (1 / (np.linalg.norm(r_i))) * r_i[0]
        return n

    def update_stats(self, t):
        self.t = t
        self.position = self.get_position(t)
        self.velocity = self.get_velocity(t)
        self.acceleration = self.get_acceleration(t)
        self.tang = self.get_tangential(t)
        self.norm = self.get_normal(t)


def init():
    # set up ax
    ax.set_xlim(-12000, 12000)
    ax.set_ylim(-12000, 12000)
    ax.set_title("aircraft trajectory")
    ax.set_xlabel("y in meter")
    ax.set_ylabel("x in meter")
    ax.grid(True)

    # set up ax2
    ax2.set_xlim(-10, 430)
    ax2.set_ylim(0, 400)
    ax2.set_title("aircraft velocity")
    ax2.set_xlabel("r'(t) in meter")
    ax2.set_ylabel("t in meter")
    ax2.grid(True)

    # set up ax3
    ax3.set_xlim(-10, 430)
    ax3.set_ylim(-2, 12)
    ax3.set_title("aircraft acceleration")
    ax3.set_xlabel("r''(t) in m/s²")
    ax3.set_ylabel("t in meter")
    ax3.grid(True)

    # set up ax4
    ax4.set_xlim(-10, 430)
    ax4.set_ylim(-10, 10)
    ax4.set_title("r''(t)t(t)")
    ax4.set_xlabel("r''(t)t(t) in m/s²")
    ax4.set_ylabel("t in meter")
    ax4.grid(True)

    # set up ax5
    ax5.set_xlim(-10, 430)
    ax5.set_ylim(-10, 10)
    ax5.set_title("r''(t)n(t)")
    ax5.set_xlabel("r''(t)n(t) in m/s²")
    ax5.set_ylabel("t in meter")
    ax5.grid(True)

    #set up ax6
    ax6.set_title("sensor polar measurements")
    ax6.set_xlabel("time in seconds")
    ax6.set_ylabel("range in meter")
    ax6.grid(True)
    ax6.set_xlim(-24000, 24000)
    ax6.set_ylim(-24000, 24000)


    return ln, ln2, ln3, ln4, ln5, ln6, ln7, ln8, ln9, ln10


def update(frame):

    # update values
    air.update_stats(frame)
    # appending data for object trace
    xdata.append(air.position[0])
    ydata.append(air.position[1])
    ln.set_data(xdata, ydata)
    # set up line for moving object
    ln2.set_data(air.position[0], air.position[1])

    # set up time array
    time.append(frame)

    # set up line for aircraft velocity
    v.append(np.linalg.norm(air.velocity))
    ln3.set_data(time, v)

    # set up line for aircraft acceleration
    q.append(np.linalg.norm(air.acceleration))
    ln4.set_data(time, q)

    # set up line for r''(t)t(t)
    r_t.append(np.matmul(air.acceleration.T, air.tang))
    ln5.set_data(time, r_t)

    # set up line for r''(t)n(t)
    r_n.append(np.matmul(air.acceleration.T, air.norm))
    ln6.set_data(time, r_n)

    # update sensor data
    sensor.update_stats(frame)

    # set up sensor measurements
    # cartesian
    if frame % sensor.delta_t == 0:
        cx.append(sensor.z_c[0])
        cy.append(sensor.z_c[1])
        ln9.set_data(cx, cy)

    # def polar vars
    px = 0
    py = 0
    # polar
    if frame % sensor.delta_t == 0:
        sens_time.append(frame)

        z_r = sensor.z_p[0]
        z_f = sensor.z_p[1]
        r = sensor.pos
        vec = np.array((np.cos(z_f), np.sin(z_f)))
        vec = vec.reshape((2, 1))
        polar = z_r * vec + r
        px = polar[0]
        py = polar[0]
        polarx.append(polar[0])
        polary.append(polar[1])
        #ln7.set_data(polarx, polary)
        #ln10.set_data(sens_time, zs)


    # set up sensor position
    ln8.set_data(sensor.pos[0], sensor.pos[1])

    # plot vectors as quiver
    # plot tangential vector
    Q1 = ax.quiver(air.position[0], air.position[1], air.tang[0], air.tang[1], units='xy',
                         scale=0.0005, color='g')
    qk1 = ax.quiverkey(Q1, 0.9, 0.9, 2, r'$tangential vector$', labelpos='E',
                             coordinates='figure')
    # plot normal vector
    Q2 = ax.quiver(air.position[0], air.position[1], air.norm[0], air.norm[1], units='xy',
                         scale=0.0005, color='y')
    qk2 = ax.quiverkey(Q1, 0.9, 0.9, 2, r'$normal vector$', labelpos='E',
                             coordinates='figure')

    # plot polar measurements
    Q3 = ax6.quiver(sensor.pos[0], sensor.pos[1], polarx, polary, units='xy',
                   scale=1, color='y')
    qk3 = ax6.quiverkey(Q1, 0.9, 0.9, 2, 'polar vector', labelpos='E',
                       coordinates='figure')
    return ln, ln2, ln3, ln4, ln5, ln6, ln7, ln8, ln9, ln10, Q1, Q2, Q3,


if __name__ == "__main__":
    T = 419
    # set up objects
    air = Aircraft(300, 9, 0)
    sensor_position = np.array((8000, 6500))
    sensor_position = sensor_position.reshape((2, 1))
    # o.2° is 0.00349066 radiant
    sensor = sen.Sensor(50, 20, 0.00349066, 5, sensor_position, air)

    # set up figure and subplots
    font = {'size': 9}
    matplotlib.rc('font', **font)
    fig = figure(num=0, figsize=(12, 8))  # , dpi = 100)
    fig.suptitle("ground truth generator", fontsize=12)
    ax = plt.subplot2grid((3, 3), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
    ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
    ax4 = plt.subplot2grid((3, 3), (2, 0))
    ax5 = plt.subplot2grid((3, 3), (2, 1))
    ax6 = plt.subplot2grid((3, 3), (0, 2))

    # set up arrays to save data of plots
    xdata, ydata = [], []
    v = []
    q = []
    r_t = []
    r_n = []
    time = []
    # time for sensor as depends on rate
    sens_time = []
    # for cartesian measurements
    cx, cy = [], []
    # for polar measurements
    polarx, polary = [], []

    # set up lines to plot
    # object trace
    ln, = ax.plot([], [])
    # moving object
    ln2, = ax.plot([], [], 'ro')
    # velocity
    ln3, = ax2.plot([], [])
    # acceleration
    ln4, = ax3.plot([], [])
    # vector products with t(t) & n(t)
    ln5, = ax4.plot([], [])
    ln6, = ax5.plot([], [])
    # sensor measurements polar range
    ln7, = ax6.plot([], [], 'bo', label="range measurement")
    # sensor position
    ln8, = ax.plot([], [], 'yo', label="sensor position")
    # cartesian
    ln9, = ax.plot([], [], 'ro', label="cartesian measurement")
    # sensor measurements polar winkel
    ln10, = ax6.plot([], [], 'ro', label="azimuth")


    # plot animation
    ani = FuncAnimation(fig, update, frames=T,
                        init_func=init, interval=50,  blit=True, repeat=False)

    ax.legend()
    ax6.legend()
    plt.tight_layout()
    plt.show()
