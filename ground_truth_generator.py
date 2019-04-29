import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
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

class AnimatedPlot(object):
    def __init__(self, ax1, ax2, air):
        # init axes
        self.ax1 = ax1
        self.ax2 = ax2
        self.air = air

        # set axis limits and labels
        self.ax1.set_xlim(-12000, 12000)
        self.ax1.set_ylim(-12000, 12000)
        self.ax1.set_xlabel("X in m")
        self.ax1.set_ylabel("Y in m")
        self.ax1.set_title("aircraft trajectory")

        # plot lines
        self.line, = ax1.plot([], [], 'bo')

    def init(self):
        self.line.set_data([], [])
        return self.line,

    def __call__(self, i):
        if i == 0:
            return self.init()

        # get position of aircraft
        self.air.update_stats(i)
        x = self.air.position[0]
        y = self.air.position[1]
        self.line.set_data(x, y)

        # plot tangential vector
        Q1 = self.ax1.quiver(self.air.position[0], self.air.position[1], self.air.tang[0], self.air.tang[1], units='xy', scale=0.0005, color='g')
        qk1 = self.ax1.quiverkey(Q1, 0.9, 0.9, 2, r'$tangential vector$', labelpos='E',
                                 coordinates='figure')
        # plot normal vector
        Q2 = self.ax1.quiver(self.air.position[0], self.air.position[1], self.air.norm[0], self.air.norm[1], units='xy', scale=0.0005, color='r')
        qk2 = self.ax1.quiverkey(Q1, 0.9, 0.9, 2, r'$normal vector$', labelpos='E',
                                 coordinates='figure')

        return self.line, Q1, Q2

def main():
    # period T is 400/3 * Pi = 418.879 ...
    T = 419
    air = Aircraft(300, 9, 0)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    ap = AnimatedPlot(ax1, air)
    anim = FuncAnimation(fig, ap, frames=T, init_func=ap.init,
                         interval=10, blit=True)

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
