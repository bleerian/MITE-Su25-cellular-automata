# spinodal_decomp.py

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML  # Added for Colab compatibility

class SpinodalBase:
    def __init__(self, n, nstep, d, AA=1.3):
        self.n = n
        self.nstep = int(nstep)
        self.d = d
        self.AA = AA

        # Default initial condition: random values around 0
        self.eta = 0.1 * (2 * np.random.rand(n, n) - 1)

        ### Alternate initial profiles (uncomment to use) ###

        ## Non-parity concentration profile
        # self.eta = 0.1 * (2 * np.random.rand(n, n) - 4)

        ## Seed in the middle
        # self.eta = 0.001 * (2 * np.random.rand(n, n) - 1)
        # mid = int(n / 2)
        # self.eta[mid, mid] = 1.0

    def pbc_neighbors(self, field):
        A1 = (np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) +
              np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1))
        A2 = (np.roll(np.roll(field, 1, axis=0), 1, axis=1) +
              np.roll(np.roll(field, -1, axis=0), 1, axis=1) +
              np.roll(np.roll(field, 1, axis=0), -1, axis=1) +
              np.roll(np.roll(field, -1, axis=0), -1, axis=1))
        return A1, A2

    def animate(self, interval=100):
        fig, ax = plt.subplots()
        im = ax.imshow(self.eta, cmap='bwr', vmin=-1, vmax=1, interpolation='none')
        plt.colorbar(im, ax=ax)
        plt.title(self.title)

        def update_plot(frame):
            self.step()
            im.set_data(self.eta.copy())
            return [im]

        anim = FuncAnimation(fig, update_plot, frames=range(self.nstep), interval=interval, blit=False)
        plt.close(fig)  # Prevents static image from rendering
        return HTML(anim.to_jshtml())  # Inline HTML animation for Colab
        
    def plot(self):
        plt.figure(figsize=(5, 5))
        plt.imshow(self.eta, cmap='bwr', vmin=-1, vmax=1, interpolation='none')
        plt.colorbar(label='η (concentration)')
        plt.title(self.title)
        plt.axis('off')
        plt.show()


class SpinodalNonConserved(SpinodalBase):
    def __init__(self, n, nstep, d, AA=1.3):
        super().__init__(n, nstep, d, AA)
        self.title = "Spinodal Decomposition (Non-Conserved)"

    def step(self):
        A1, A2 = self.pbc_neighbors(self.eta)
        etat = self.AA * np.tanh(self.eta) + self.d * (A1 / 6 + A2 / 12 - self.eta)
        self.eta = etat.copy()


class SpinodalConserved(SpinodalBase):
    def __init__(self, n, nstep, d, AA=1.3):
        super().__init__(n, nstep, d, AA)
        self.title = "Spinodal Decomposition (Conserved)"

    def step(self):
        A1, A2 = self.pbc_neighbors(self.eta)
        st = self.AA * np.tanh(self.eta) + self.d * (A1 / 6 + A2 / 12 - self.eta)
        B1, B2 = self.pbc_neighbors(st)
        stt = st - (B1 - A1) / 6 - (B2 - A2) / 12
        self.eta = stt.copy()
