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

def init():
    ax.set_xlim(-12000, 12000)
    ax.set_ylim(-12000, 12000)
    ax.set_title("aircraft trajectory")
    ax.set_xlabel("y in meter")
    ax.set_ylabel("x in meter")
    ax.grid(True)
    return ln, ln2,

def update(frame):
    air.update_stats(frame)
    xdata.append(air.position[0])
    ydata.append(air.position[1])
    ln.set_data(xdata, ydata)
    ln2.set_data(air.position[0], air.position[1])

    #plot vectors as quiver
    Q1 = ax.quiver(air.position[0], air.position[1], air.tang[0], air.tang[1], units='xy',
                         scale=0.0005, color='g')
    qk1 = ax.quiverkey(Q1, 0.9, 0.9, 2, r'$tangential vector$', labelpos='E',
                             coordinates='figure')
    # plot normal vector
    Q2 = ax.quiver(air.position[0], air.position[1], air.norm[0], air.norm[1], units='xy',
                         scale=0.0005, color='y')
    qk2 = ax.quiverkey(Q1, 0.9, 0.9, 2, r'$normal vector$', labelpos='E',
                             coordinates='figure')
    return ln, ln2, Q1, Q2,

def main():
  pass

if __name__ == "__main__":
    T = 419
    air = Aircraft(300, 9, 0)

    fig, ax = plt.subplots()
    xdata, ydata = [], []
    ln, = plt.plot([], [])
    ln2, = plt.plot([], [], 'ro')
    ani = FuncAnimation(fig, update, frames=T,
                        init_func=init, interval=10,  blit=True)
    plt.show()
    #main()