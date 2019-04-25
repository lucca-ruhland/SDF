import numpy as np
import matplotlib.pyplot as plt
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
        t = np.zeros((2, 1))
        r_i = self.get_velocity(t)
        print(r_i.shape)
        t[0] = 1/(np.linalg.norm(r_i)) * r_i[0]
        t[1] = 1/(np.linalg.norm(r_i)) * r_i[1]
        print(t.shape)
        return t

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
        #self.tang = self.get_tangential(t)
        self.norm = self.get_normal(t)

class AnimatedPlot(object):
    def __init__(self, ax, air):
        self.ax = ax
        self.air = air
        self.ax.set_xlim(-12000, 12000)
        self.ax.set_ylim(-12000, 12000)
        self.ax.set_title("aircraft tracerory")
        self.line, = ax.plot([], [], 'bo')

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
        return self.line,

def main():
    air = Aircraft(300, 9, 0)
    fig, ax = plt.subplots()
    ap = AnimatedPlot(ax, air)
    anim = FuncAnimation(fig, ap, frames=10000, init_func=ap.init,
                         interval=10, blit=True)
    plt.show()

if __name__ == "__main__":
    main()
