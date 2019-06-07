from matplotlib.pylab import *
from matplotlib.animation import FuncAnimation
import sensor_simulator as sen
import aircraft as airc
from matplotlib.lines import Line2D
import kalman_filter as kf


class Animation(object):

    def __init__(self, ax, ax2, ax3, ax4, ax5, ax6, air, sensor, *args):
        """Setting up all local variables and calculate plot data"""
        # set up objects
        self.ax = ax
        self.ax2 = ax2
        self.ax3 = ax3
        self.ax4 = ax4
        self.ax5 = ax5
        self.ax6 = ax6
        self.air = air
        self.sensor = sensor

        # variable number of sensors
        self.sensors = args
        self.sen_num = 0
        for arg in args:
            if isinstance(arg, sen.Sensor):
                self.sen_num += 1

        # set up axis and lines
        # set up ax
        self.ax.set_xlim(-12000, 12000)
        self.ax.set_ylim(-12000, 12000)
        ax.set_title("aircraft trajectory")
        self.ax.set_xlabel("y in meter")
        self.ax.set_ylabel("x in meter")
        ax.grid(True)
        # object trace
        self.ln = Line2D([], [], color='b')
        # moving object
        self.ln2 = Line2D([], [], color='r', marker='o')
        # polar measurements
        self.ln7 = Line2D([], [], color='g', marker='^', linewidth=0)
        # sensor position
        self.ln8 = Line2D([], [], color='r', marker='o')
        # cartesian
        self.ln9 = Line2D([], [], color='r', marker='x', linewidth=0)
        # prediction
        self.ln10 = Line2D([], [], color='r', marker='o', linewidth=0)
        # polar measurement for each sensor
        self.ln11 = Line2D([], [], color='g', marker='x', linewidth=0)
        # fused polar measurements
        self.ln12 = Line2D([], [], color='r', marker='^', linewidth=0)

        ax.add_line(self.ln)
        ax.add_line(self.ln2)
        ax.add_line(self.ln7)
        ax.add_line(self.ln8)
        ax.add_line(self.ln9)
        ax.add_line(self.ln10)
        ax.add_line(self.ln11)
        ax.add_line(self.ln12)

        # set up ax2
        self.ax2.set_xlim(-10, 430)
        self.ax2.set_ylim(0, 400)
        ax2.set_title("aircraft velocity")
        self.ax2.set_xlabel("r'(t) in meter")
        self.ax2.set_ylabel("t in meter")
        ax2.grid(True)
        # velocity
        self.ln3 = Line2D([], [], color='b')
        ax2.add_line(self.ln3)

        # set up ax3
        self.ax3.set_xlim(-10, 430)
        self.ax3.set_ylim(-1, 10)
        ax3.set_title("aircraft acceleration")
        self.ax3.set_xlabel("r''(t) in m/s^2")
        self.ax3.set_ylabel("t in meter")
        ax3.grid(True)
        # acceleration
        self.ln4 = Line2D([], [], color='b')
        ax3.add_line(self.ln4)

        # set up ax4
        self.ax4.set_xlim(-10, 430)
        self.ax4.set_ylim(-10, 10)
        ax4.set_title("r''(t)t(t)")
        self.ax4.set_xlabel("r''(t)t(t) in m/s^2")
        self.ax4.set_ylabel("t in meter")
        ax4.grid(True)
        # vector products with t(t) & n(t)
        self.ln5 = Line2D([], [], color='b')
        ax4.add_line(self.ln5)

        # set up ax5
        self.ax5.set_xlim(-10, 430)
        self.ax5.set_ylim(-10, 10)
        ax5.set_title("r''(t)n(t)")
        self.ax5.set_xlabel("r''(t)n(t) in m/s^2")
        self.ax5.set_ylabel("t in meter")
        ax5.grid(True)
        self.ln6 = Line2D([], [], color='b')
        ax5.add_line(self.ln6)

        # set up ax6
        ax6.set_title("sensor polar measurements")
        self.ax6.set_xlabel("time in seconds")
        self.ax6.set_ylabel("range in meter")
        ax6.grid(True)
        self.ax6.set_xlim(-24000, 24000)
        self.ax6.set_ylim(-24000, 24000)

        # add kalman object
        self.kalman_filter = kf.kalman(air, 5, sensor, sensor2, sensor3, sensor4)

        # set up data for lines
        # self.t = np.linspace(0, 420, 420)
        self.t = np.arange(0, 420, 1)
        self.v_abs = np.array([np.linalg.norm(air.get_velocity(i)) for i in self.t])
        self.q_abs = np.array([np.linalg.norm(air.get_acceleration(i)) for i in self.t])
        self.tang = np.array([np.dot(air.get_acceleration(i), air.get_tangential(i)) for i in self.t])
        self.norm = np.array([np.dot(air.get_acceleration(i), air.get_normal(i)) for i in self.t])

        self.cartesian_x = np.zeros(420)
        self.cartesian_y = np.zeros(420)
        self.polar_x = np.zeros(420)
        self.polar_y = np.zeros(420)

        # prediction
        self.prediction_x = np.zeros(420)
        self.prediction_y = np.zeros(420)

        # polar measurement transformed to cartesian for each sensor
        self.polar_meas_x = np.zeros((self.sen_num, 420))
        self.polar_meas_y = np.zeros((self.sen_num, 420))

        # fused polar measurements transformed to cartesian
        self.polar_fused_x = np.zeros(420)
        self.polar_fused_y = np.zeros(420)

        # calculate sensor data for each instance delta t
        for i in self.t:
            if i % self.sensor.delta_t == 0 or i == 0:
                self.cartesian_x[i] = self.sensor.cartesian(i)[0]
                self.cartesian_y[i] = self.sensor.cartesian(i)[1]
                self.polar_x[i] = self.sensor.range(i) * np.cos(self.sensor.azimuth(i)) + self.sensor.pos[0]
                self.polar_y[i] = self.sensor.range(i) * np.sin(self.sensor.azimuth(i)) + self.sensor.pos[1]

                # get fused polar measurements
                self.kalman_filter.update_polar(i)
                self.polar_fused_x[i] = self.kalman_filter.z[0]
                self.polar_fused_y[i] = self.kalman_filter.z[1]

                # get prediction after filtering
                self.kalman_filter.update_polar(i)
                self.prediction_x[i] = self.kalman_filter.x[0]
                self.prediction_y[i] = self.kalman_filter.x[1]

                # get measurement for each sensor
                for arg in self.sensors:
                    j = 0
                    if isinstance(arg, sen.Sensor):
                        arg.update_stats(i)
                        z = arg.z_r * np.array([np.cos(arg.z_az), np.sin(arg.z_az)]).reshape((2, 1)) + arg.pos
                        self.polar_meas_x[j, i] = z[0]
                        self.polar_meas_y[j, i] = z[1]
                        j = j +1

            # fill up array with existing values
            else:
                self.cartesian_x[i] = self.cartesian_x[i-1]
                self.cartesian_y[i] = self.cartesian_y[i-1]
                self.polar_x[i] = self.polar_x[i-1]
                self.polar_y[i] = self.polar_y[i-1]
                self.kalman_filter.update_polar(i)
                self.prediction_x[i] = self.kalman_filter.x[0]
                self.prediction_y[i] = self.kalman_filter.x[1]

                # fill up measurements of each sensor
                for arg in self.sensors:
                    j = 0
                    if isinstance(arg, sen.Sensor):
                        arg.update_stats(i)
                        self.polar_meas_x[j, i] = self.polar_meas_x[j, i-1]
                        self.polar_meas_y[j, i] = self.polar_meas_y[j, i-1]

        # calculate position
        self.pos_x = np.array([self.air.get_position(i)[0] for i in self.t])
        self.pos_y = np.array([self.air.get_position(i)[1] for i in self.t])

    def init_plot(self):
        """Setting all initial values for the data plots"""
        lines = [self.ln, self.ln2, self.ln3, self.ln4, self.ln5, self.ln6, self.ln7, self.ln8, self.ln9, self.ln10,
                 self.ln11, self.ln12]
        for l in lines:
            l.set_data([], [])
        # set sensor position
        self.ax.plot(self.sensor.pos[0], self.sensor.pos[1], 'ro')
        self.ax6.plot(self.sensor.pos[0], self.sensor.pos[1], 'ro')

        # set up sensor position for variable num of sensors
        for arg in self.sensors:
            if isinstance(arg, sen.Sensor):
                self.ax.plot(arg.pos[0], arg.pos[1], 'ro')

        return lines

    def __call__(self, i):
        """Updates for each frame the animation"""
        self.air.update_stats(i)
        x = self.air.position[0]
        y = self.air.position[1]
        # update aircraft trajectory
        self.ln.set_data(self.pos_x[:i], self.pos_y[:i])
        # update aircraft position
        self.ln2.set_data(x, y)

        # update aircraft velocity
        self.ln3.set_data(self.t[:i], self.v_abs[:i])

        # update aircraft acceleration
        self.ln4.set_data(self.t[:i], self.q_abs[:i])

        # update rt
        self.ln5.set_data(self.t[:i], self.norm[:i])

        # update rn
        self.ln6.set_data(self.t[:i], self.tang[:i])

        # update cartesian measurements
        # if i % self.sensor.delta_t == 0:
        # self.ln9.set_data(self.cartesian_x[:i], self.cartesian_y[:i])
        # self.ln7.set_data(self.polar_x[:i], self.polar_y[:i])

        # kalman plot prediction after filter
        self.ln10.set_data(self.prediction_x[i-10:i], self.prediction_y[i-10:i])

        # kalman plot measurements for each sensor
        self.ln11.set_data(self.polar_meas_x[:, :i], self.polar_meas_y[:, :i])

        # kalman plot fused measurement
        self.ln12.set_data(self.polar_fused_x[:i], self.polar_fused_y[:i])

        # plot quiver for object
        q_tang = self.ax.quiver(x, y, self.air.tang[0], self.air.tang[1], pivot='tail', color='black', width=0.004, angles='xy', scale=25)
        # plot normal vector
        q_norm = self.ax.quiver(x, y, self.air.norm[0], self.air.norm[1], pivot='tail', color='black', width=0.004, scale=25)
        # plot polar measurements
        q_polar = ax.quiver(self.sensor.pos[0], self.sensor.pos[1], self.polar_x[i] - self.sensor.pos[0], self.polar_y[i]- self.sensor.pos[1], pivot='tail',
                             color='green', angles='xy', units='xy', scale=1, scale_units='xy', width=70)

        artists = [self.ln, self.ln2, self.ln3, self.ln4, self.ln5, self.ln6, self.ln7, self.ln9, self.ln10, self.ln11,
                   self.ln12, q_norm, q_tang, q_polar]

        # plot quivers for each sensor
        for arg in self.sensors:
            if isinstance(arg, sen.Sensor):
                arg.update_stats(i)
                # artists.append(ax.quiver(arg.pos[0], arg.pos[1], self.polar_x[i] - arg.pos[0], self.polar_y[i] - arg.pos[1], pivot='tail', color='green', angles='xy', units='xy', scale=1, scale_units='xy', width=70))
                # all vectors pointing to their own measurement
                z = arg.z_r * np.array([np.cos(arg.z_az), np.sin(arg.z_az)]).reshape((2, 1)) + arg.pos
                artists.append(ax.quiver(arg.pos[0], arg.pos[1], z[0] - arg.pos[0], z[1] - arg.pos[1], pivot='tail', color='green', angles='xy', units='xy', scale=1, scale_units='xy', width=70))

        return artists


if __name__ == "__main__":

    T = 420
    # set up objects
    air = airc.Aircraft(300, 9, 0)
    sensor_position = np.array((-11000, -11000)).reshape((2, 1))
    sensor_position2 = np.array((-11000, 11000)).reshape((2, 1))
    sensor_position3 = np.array((11000, -11000)).reshape((2, 1))
    sensor_position4 = np.array((11000, 11000)).reshape((2, 1))
    # o.2 grad is 0.00349066 radiant
    sensor = sen.Sensor(50, 20, 0.00349066, 5, sensor_position, air)
    sensor2 = sen.Sensor(50, 20, 0.00349066, 5, sensor_position2, air)
    sensor3 = sen.Sensor(50, 20, 0.00349066, 5, sensor_position3, air)
    sensor4 = sen.Sensor(50, 20, 0.00349066, 5, sensor_position4, air)

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
    ani = Animation(ax, ax2, ax3, ax4, ax5, ax6, air, sensor, sensor2, sensor3, sensor4)
    animation = FuncAnimation(fig, ani, frames=T, init_func=ani.init_plot, interval=50,  blit=True, repeat=True)

    # ax.legend()
    # ax6.legend()
    plt.tight_layout()
    plt.show()
