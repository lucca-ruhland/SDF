from matplotlib.pylab import *
from matplotlib.animation import FuncAnimation
from sensor_simulator import Sensor
from aircraft import Aircraft
from matplotlib.lines import Line2D
from kalman_filter import KalmanFilter
import matplotlib.gridspec as gridspec


class Animation(object):

    def __init__(self, ax1, ax2, ax3, ax4, ax5, ax6, air, *args):
        """Setting up all local variables and calculate plot data"""
        # set up objects
        self.ax1 = ax1
        self.ax2 = ax2
        self.ax3 = ax3
        self.ax4 = ax4
        self.ax5 = ax5
        self.ax6 = ax6
        # set aircraft object
        self.air = air
        # add kalman object
        self.kalman_filter = KalmanFilter(air, 5, *args)

        # variable number of sensors
        self.sensors = args
        self.sen_num = 0
        self.delta_t = 0
        for arg in args:
            if isinstance(arg, Sensor):
                self.delta_t = arg.delta_t
                self.sen_num += 1

        # set up lines
        # object trace
        self.ln_trace = Line2D([], [], color='b')
        # moving object
        self.ln_object = Line2D([], [], color='k', marker='o')
        # prediction
        self.ln_prediction = Line2D([], [], color='m', marker='o', linewidth=0)
        # polar measurement for each sensor
        self.ln_meas = Line2D([], [], color='g', marker='x', linewidth=0)
        # fused polar measurements
        self.ln_fused = Line2D([], [], color='r', marker='^', linewidth=0)
        # acceleration
        self.ln_acc = Line2D([], [], color='b')
        # velocity
        self.ln_vel = Line2D([], [], color='b')
        # vector products with r''(t) & t(t)
        self.ln_vec_t = Line2D([], [], color='b')
        # vector products with r''(t) & n(t)
        self.ln_vec_n = Line2D([], [], color='b')
        # difference between filtered prediction and real position
        self.ln_diff = Line2D([], [], color='b')

        # set up ax1
        self.ax1.set_xlim(-12000, 12000)
        self.ax1.set_ylim(-12000, 12000)
        ax1.set_title("aircraft trajectory")
        self.ax1.set_xlabel("y in meter")
        self.ax1.set_ylabel("x in meter")
        ax1.grid(True)

        ax1.add_line(self.ln_trace)
        ax1.add_line(self.ln_object)
        ax1.add_line(self.ln_prediction)
        ax1.add_line(self.ln_meas)
        ax1.add_line(self.ln_fused)
        # set up legend
        legend_lines = [self.ln_prediction, self.ln_meas, self.ln_fused]
        box = ax1.get_position()
        ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                         box.width, box.height * 0.9])

        # Put a legend below current axis
        ax1.legend(legend_lines, ['filtered prediction', 'polar measurements', 'fused measurements'], loc='upper center',
                  bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=5)

        # set up ax2
        self.ax2.set_xlim(-10, 430)
        self.ax2.set_ylim(0, 400)
        ax2.set_title("aircraft velocity")
        self.ax2.set_xlabel("r'(t) in meter")
        self.ax2.set_ylabel("t in meter")
        ax2.grid(True)
        ax2.add_line(self.ln_vel)

        # set up ax3
        self.ax3.set_xlim(-10, 430)
        self.ax3.set_ylim(-1, 10)
        ax3.set_title("aircraft acceleration")
        self.ax3.set_xlabel("r''(t) in m/s^2")
        self.ax3.set_ylabel("t in meter")
        ax3.grid(True)
        ax3.add_line(self.ln_acc)

        # set up ax4
        self.ax4.set_xlim(-10, 430)
        self.ax4.set_ylim(-10, 10)
        ax4.set_title("r''(t)t(t)")
        self.ax4.set_xlabel("r''(t)t(t) in m/s^2")
        self.ax4.set_ylabel("t in meter")
        ax4.grid(True)
        ax4.add_line(self.ln_vec_t)

        # set up ax5
        self.ax5.set_xlim(-10, 430)
        self.ax5.set_ylim(-10, 10)
        ax5.set_title("r''(t)n(t)")
        self.ax5.set_xlabel("r''(t)n(t) in m/s^2")
        self.ax5.set_ylabel("t in meter")
        ax5.grid(True)
        ax5.add_line(self.ln_vec_n)

        # set up ax6
        ax6.set_title("Difference between filtered prediction and real object position")
        self.ax6.set_xlabel("time in seconds")
        self.ax6.set_ylabel("distance in meter")
        ax6.grid(True)
        self.ax6.set_xlim(-10, 430)
        self.ax6.set_ylim(-10, 1000)
        ax6.add_line(self.ln_diff)

        # set up data for lines
        # time array
        self.t = np.arange(0, 420, 1)
        # velocity of aircraft
        self.v_abs = np.array([np.linalg.norm(air.get_velocity(i)) for i in self.t])
        # acceleration of aircraft
        self.q_abs = np.array([np.linalg.norm(air.get_acceleration(i)) for i in self.t])
        # tangential vector of aircraft
        self.tang = np.array([np.dot(air.get_acceleration(i), air.get_tangential(i)) for i in self.t])
        # normal vector of aircraft
        self.norm = np.array([np.dot(air.get_acceleration(i), air.get_normal(i)) for i in self.t])

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
            if i % self.delta_t == 0 or i == 0:
                # get fused polar measurements
                self.kalman_filter.update_polar(i)
                self.polar_fused_x[i] = self.kalman_filter.z[0]
                self.polar_fused_y[i] = self.kalman_filter.z[1]

                # get prediction after filtering
                self.kalman_filter.update_polar(i)
                self.prediction_x[i] = self.kalman_filter.x[0]
                self.prediction_y[i] = self.kalman_filter.x[1]

                j = 0
                # get measurement for each sensor
                for arg in self.sensors:
                    if isinstance(arg, Sensor):
                        arg.update_stats(i)
                        z = arg.z_r * np.array([np.cos(arg.z_az), np.sin(arg.z_az)]).reshape((2, 1)) + arg.pos
                        self.polar_meas_x[j, i] = z[0]
                        self.polar_meas_y[j, i] = z[1]
                        j = j + 1

            # fill up array with existing values
            else:
                self.kalman_filter.update_polar(i)
                self.prediction_x[i] = self.kalman_filter.x[0]
                self.prediction_y[i] = self.kalman_filter.x[1]

                # fill up measurements of each sensor
                j = 0
                for arg in self.sensors:
                    if isinstance(arg, Sensor):
                        # arg.update_stats(i)
                        self.polar_meas_x[j, i] = self.polar_meas_x[j, i-1]
                        self.polar_meas_y[j, i] = self.polar_meas_y[j, i-1]
                        j = j + 1

        # calculate position
        self.pos_x = np.array([self.air.get_position(i)[0] for i in self.t])
        self.pos_y = np.array([self.air.get_position(i)[1] for i in self.t])

        # calculate difference
        self.diff_x = np.array([self.prediction_x[i] - self.pos_x[i] for i in self.t])
        self.diff_y = np.array([self.prediction_y[i] - self.pos_y[i] for i in self.t])
        self.diff_abs = np.array([np.linalg.norm((self.diff_x[i], self.diff_y[i])) for i in self.t])
        # set max y of ax6
        self.ax6.set_ylim(-10, np.amax(self.diff_abs) * 1.2)

    def init_plot(self):
        """Setting all initial values for the data plots"""
        lines = [self.ln_trace, self.ln_object, self.ln_vel, self.ln_acc, self.ln_vec_t, self.ln_vec_n,
                 self.ln_prediction, self.ln_meas, self.ln_fused, self.ln_diff]
        for l in lines:
            l.set_data([], [])

        # set up sensor position for variable num of sensors
        for arg in self.sensors:
            if isinstance(arg, Sensor):
                self.ax1.plot(arg.pos[0], arg.pos[1], 'go')

        return lines

    def __call__(self, i):
        """Updates for each frame the animation"""
        self.air.update_stats(i)
        x = self.air.position[0]
        y = self.air.position[1]
        # update aircraft trajectory
        self.ln_trace.set_data(self.pos_x[:i], self.pos_y[:i])
        # update aircraft position
        self.ln_object.set_data(x, y)

        # update aircraft velocity
        self.ln_vel.set_data(self.t[:i], self.v_abs[:i])

        # update aircraft acceleration
        self.ln_acc.set_data(self.t[:i], self.q_abs[:i])

        # update rt
        self.ln_vec_t.set_data(self.t[:i], self.norm[:i])

        # update rn
        self.ln_vec_n.set_data(self.t[:i], self.tang[:i])

        # kalman plot prediction after filter
        self.ln_prediction.set_data(self.prediction_x[i-15:i], self.prediction_y[i-15:i])

        # kalman plot measurements for each sensor
        self.ln_meas.set_data(self.polar_meas_x[:, :i], self.polar_meas_y[:, :i])

        # kalman plot fused measurement
        self.ln_fused.set_data(self.polar_fused_x[:i], self.polar_fused_y[:i])

        # kalman plot difference between filtered prediction and real object position
        self.ln_diff.set_data(self.t[:i], self.diff_abs[:i])

        # plot quiver for object
        q_tang = self.ax1.quiver(x, y, self.air.tang[0], self.air.tang[1], pivot='tail', color='black', width=0.004,
                                angles='xy', scale=35)
        # plot normal vector
        q_norm = self.ax1.quiver(x, y, self.air.norm[0], self.air.norm[1], pivot='tail', color='black', width=0.004,
                                scale=35)

        artists = [self.ln_trace, self.ln_object, self.ln_vel, self.ln_acc, self.ln_vec_t, self.ln_vec_n,
                   self.ln_prediction, self.ln_meas, self.ln_fused, self.ln_diff, q_norm, q_tang]

        # plot quivers for each sensor
        j = 0
        for arg in self.sensors:
            if isinstance(arg, Sensor):
                arg.update_stats(i)
                # all vectors pointing to their own measurement
                # z = arg.z_r * np.array([np.cos(arg.z_az), np.sin(arg.z_az)]).reshape((2, 1)) + arg.pos
                z = np.array([self.polar_meas_x[j, i], self.polar_meas_y[j, i]])
                artists.append(ax1.quiver(arg.pos[0], arg.pos[1], z[0] - arg.pos[0], z[1] - arg.pos[1], pivot='tail',
                                         color='green', angles='xy', units='xy', scale=1, scale_units='xy', width=70))
                j = j + 1

        return artists


if __name__ == "__main__":
    # Testing Animation with Kalman Filter
    T = 420
    # set up objects
    air = Aircraft(300, 9, 0)
    sensor_position = np.array((-11000, -11000)).reshape((2, 1))
    sensor_position2 = np.array((-11000, 11000)).reshape((2, 1))
    sensor_position3 = np.array((11000, -11000)).reshape((2, 1))
    sensor_position4 = np.array((11000, 11000)).reshape((2, 1))
    # even more sensor test
    # sensor_position5 = np.array((-1100, -11000)).reshape((2, 1))
    # sensor_position6 = np.array((-4000, 7000)).reshape((2, 1))
    # sensor_position7 = np.array((2000, -9000)).reshape((2, 1))
    # sensor_position8 = np.array((100, 980)).reshape((2, 1))
    # o.2 grad is 0.00349066 radiant
    sensor = Sensor(50, 20, 0.00349066, 5, sensor_position, air)
    sensor2 = Sensor(50, 20, 0.00349066, 5, sensor_position2, air)
    sensor3 = Sensor(50, 20, 0.00349066, 5, sensor_position3, air)
    sensor4 = Sensor(50, 20, 0.00349066, 5, sensor_position4, air)

    # sensor5 = Sensor(50, 20, 0.00349066, 5, sensor_position5, air)
    # sensor6 = Sensor(50, 20, 0.00349066, 5, sensor_position6, air)
    # sensor7 = Sensor(50, 20, 0.00349066, 5, sensor_position7, air)
    # sensor8 = Sensor(50, 20, 0.00349066, 5, sensor_position8, air)

    # set up figure and subplots
    font = {'size': 9}
    matplotlib.rc('font', **font)
    fig = figure(num=0, figsize=(16, 9))
    fig.suptitle("ground truth generator", fontsize=12)
    gs1 = gridspec.GridSpec(3, 3)
    gs1.update(left=0.05, right=0.48, wspace=0.05)
    ax1 = plt.subplot(gs1[:2, :])
    ax6 = plt.subplot(gs1[2, :])

    gs2 = gridspec.GridSpec(3, 3)
    gs2.update(left=0.55, right=0.98, hspace=0.5)
    ax2 = plt.subplot(gs2[1, :2])
    ax3 = plt.subplot(gs2[2, :])
    ax4 = plt.subplot(gs2[0, :2])
    ax5 = plt.subplot(gs2[:-1, -1])

    # plot animation
    ani = Animation(ax1, ax2, ax3, ax4, ax5, ax6, air, sensor, sensor2, sensor3, sensor4)
    animation = FuncAnimation(fig, ani, frames=T, init_func=ani.init_plot, interval=60,  blit=True, repeat=True)

    # plt.tight_layout()
    plt.show()
