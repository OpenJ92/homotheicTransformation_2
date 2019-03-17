import numpy as np
from class_observer import observer
from class_elipsoid import elipsiod
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class homotheicTransform():
    def __init__(self, obs, el, light_speed, time_intervals):
        self.el = el
        self.obs = obs
        self.ls = light_speed
        self.ttu = self.tensor_time_unit(time_intervals)

    def y_hat(self):
        A = -1*(self.el.shape - self.obs.location)
        B = np.apply_along_axis(lambda x: x / np.linalg.norm(x), 1, A)
        return B

    def tensor_time_unit(self, time_intervals):
        A = []
        for i in np.flip(np.linspace(0, self.el.dt*time_intervals, time_intervals)):
            A.append(np.stack([self.el.shape,
                     self.y_hat(),
                     (self.el.shape + (i * self.ls * self.y_hat()))]))
            self.el.translate_shape()
            self.el.rotate_shape(np.array([1/np.sqrt(2), 0, 1/np.sqrt(2)]), np.pi/25)
        B = np.stack(A)
        C = np.einsum('ijkl -> klji', B)
        return C

    def push_ttu(self):
        self.ttu[:, :, 2, :] += self.el.dt * self.ttu[:, :, 1, :]

    def mask_time_interval(self, exp, var):
        get_transform = self.ttu[:, :, 2, :]
        f = lambda x: exp - var < np.linalg.norm(self.obs.location - x) < exp + var
        mask_transform = np.apply_along_axis(f, 1, get_transform)
        full_mask_transform = np.stack([mask_transform for i in range(get_transform.shape[1])], axis=1).astype('int')
        return self.ttu[:, :, 0, :] * full_mask_transform

    def plot_ttu(self, index, ax):
        xyz = [self.ttu[:, i, index, :].flatten() for i in range(0, 3)]
        ax.scatter(*xyz)

    def plot_mti(self, exp, var, ax):
        mti = self.mask_time_interval(exp, var)
        xyz = [mti[:, i, :].flatten() for i in range(0, 3)]
        ax.scatter(*xyz)
