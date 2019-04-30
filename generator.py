import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import *
import matplotlib.patches as mpatch
from mpl_toolkits.axes_grid1 import host_subplot
from matplotlib.animation import FuncAnimation

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
        r_ii[0] = self.q * np.sin(2 * w * t)
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
    #set up ax
    ax.set_xlim(-12000, 12000)
    ax.set_ylim(-12000, 12000)
    ax.set_title("aircraft trajectory")
    ax.set_xlabel("y in meter")
    ax.set_ylabel("x in meter")
    ax.grid(True)

    #set up ax2
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
    ax5.set_ylim(-8, 8)
    ax5.set_title("r''(t)n(t)")
    ax5.set_xlabel("r''(t)n(t) in m/s²")
    ax5.set_ylabel("t in meter")
    ax5.grid(True)

    return ln, ln2, ln3, ln4, ln5, ln6


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
    return ln, ln2, ln3, ln4, ln5,ln6, Q1, Q2,


def main():
  pass


if __name__ == "__main__":
    T = 419
    air = Aircraft(300, 9, 0)


    font = {'size': 9}
    matplotlib.rc('font', **font)
    fig = figure(num=0, figsize=(12, 8))  # , dpi = 100)
    fig.suptitle("ground truth generator", fontsize=12)
    ax = plt.subplot2grid((3, 3), (0, 0), colspan=2)
    ax2 = plt.subplot2grid((3, 3), (1, 0), colspan=2)
    ax3 = plt.subplot2grid((3, 3), (1, 2), rowspan=2)
    ax4 = plt.subplot2grid((3, 3), (2, 0))
    ax5 = plt.subplot2grid((3, 3), (2, 1))

    #set up arrays to save data of plots
    xdata, ydata = [], []
    v = []
    q = []
    r_t = []
    r_n = []
    time = []
    # set up lines to plot
    ln, = ax.plot([], [])
    ln2, = ax.plot([], [], 'ro')
    ln3, = ax2.plot([], [])
    ln4, = ax3.plot([], [])
    ln5, = ax4.plot([], [])
    ln6, = ax5.plot([], [])
    #plot animation
    ani = FuncAnimation(fig, update, frames=T,
                        init_func=init, interval=10,  blit=True, repeat=False)
    plt.tight_layout()
    plt.show()
    #main()