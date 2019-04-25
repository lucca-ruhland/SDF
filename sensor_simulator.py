import numpy as np

class Sensor:
    def __init__(self, sigma, delta_t, pos, state, x):
        self.sigma = sigma
        self.delta_t = delta_t
        self.pos = pos
        self.state = state
        self.x = x

    def cartesian(self):
        H = np.zeros()

def main():
    pass

if __name__ == "__main__":
    main()