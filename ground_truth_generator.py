import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
from matplotlib.animation import FuncAnimation

class Aircraft:

    v = 0
    q = 0
    t = 0
    position = np.zeros((2, 1))
    velocity = np.zeros((2, 1))
    acceleration = np.zeros((2, 1))
    tang = np.zeros((2, 1))
    norm = np.zeros((2, 1))

    def __init__(self, v, q, t):
        self.v = v
        self.q = q
        self.t = t

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
        tang[0] = 1/(np.linalg.norm(r_i)) * r_i[0]
        tang[1] = 1/(np.linalg.norm(r_i)) * r_i[1]
        return tang

    def get_normal(self, t):
        n = np.zeros((2, 1))
        r_i = self.get_velocity(t)
        n[0] = 1 / (np.linalg.norm(r_i)) * -r_i[1]
        n[1] = 1 / (np.linalg.norm(r_i)) * r_i[0]
        return n

    def update_stats(self, t):
        self.t = t
        self.position = self.get_position(t)
        self.velocity = self.get_velocity(t)
        self.acceleration = self.get_acceleration(t)
        self.tang = self.get_tangential(t)
        self.norm = self.get_normal(t)

class AnimatedPlot(object):
    def __init__(self, ax1, ax2, ax3, ax4, air):
        self.ax1 = ax1
        self.ax2 = ax2
        self.ax3 = ax3
        self.ax4 = ax4
        self.air = air
        self.ax1.set_xlim(-12000, 12000)
        self.ax1.set_ylim(-12000, 12000)
        self.ax1.set_xlabel("X in m")
        self.ax1.set_ylabel("Y in m")
        self.ax1.set_title("aircraft trajectory")
        self.line, = ax1.plot([], [], 'bo')

        self.ax2.set_title("aircraft velocity")
        self.ax3.set_title("aircraft acceleration")

    def init(self):
        self.line.set_data([], [])
        return self.line,

    def __call__(self, i):
        if i == 0:
            return self.init()

        self.air.update_stats(i)
        x = self.air.position[0]
        y = self.air.position[1]
        self.line.set_data(x, y)
        Q1 = self.ax2.quiver(*self.air.get_position(self.air.t), self.air.get_velocity(self.air.t), units='width')
        qk1 = self.ax2.quiverkey(Q1, 0.9, 0.9, 2, r'$v: \frac{m}{s}$', labelpos='E',
                                 coordinates='figure')
        Q2 = self.ax3.quiver(*self.air.get_position(self.air.t), self.air.get_acceleration(self.air.t), units='width')
        qk2 = self.ax3.quiverkey(Q1, 0.9, 0.9, 2, r'$v: \frac{m}{s^2}$', labelpos='E',
                                 coordinates='figure')

        return self.line, Q1, Q2

def main():
    # period T is 400/3 * Pi = 418.879 ...
    T = 419
    air = Aircraft(300, 9, 0)
    # fig, ax = plt.subplots()
    # ax2 = fig.add_subplot(222)
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)
    ap = AnimatedPlot(ax1, ax2, ax3, ax4, air)
    anim = FuncAnimation(fig, ap, frames=T, init_func=ap.init,
                         interval=10, blit=True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
