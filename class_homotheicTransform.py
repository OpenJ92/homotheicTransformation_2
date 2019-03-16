import numpy as np
from class_observer import observer
from class_elipsoid import elipsiod

class homotheicTransform():
    def __init__(self, obs, el, light_speed, time_intervals):
        self.el = el
        self.obs = obs
        self.ls = light_speed
        self.fig, self.ax = self.construct_graph()
        self.ttu = self.tensor_time_unit(time_intervals)

    def y_hat(self):
        A = -1*(self.el.shape - self.obs.location)
        B = np.apply_along_axis(lambda x: x / np.linalg.norm(x), 1, A)
        return B

    # is it correct to construct the positions of our photons at the time_interval*self.dt
    # in reverse. Something bizzare is occuring where no matter how long we run the simmulation
    # reverse, the light will not arrive at the observation point is the object is moving faster
    # than the 'speed of light'. Perhaps we should refactor so that we update the positions of 
    # the emmited light and store in another tensor.  I'm just not sure.
    def tensor_time_unit(self, time_intervals):
        A = []
        for i in np.linspace(0, self.el.dt*time_intervals, time_intervals):
            A.append(np.stack([self.el.shape,
                     self.y_hat(),
                     (self.el.shape + (i * self.ls * self.y_hat()))]))
            self.el.translate_shape()
            self.el.rotate_shape(np.array([1/np.sqrt(2), 0, 1/np.sqrt(2)]), np.pi/25)
        B = np.stack(A)
        C = np.einsum('ijkl -> klji', B)
        return C

    def mask_time_interval(self, exp, var):
        get_transform = self.ttu[:, :, 2, :]
        mask_transform = np.apply_along_axis(lambda x: exp - var < np.linalg.norm(self.obs.location - x) < exp + var, 1, get_transform)
        full_mask_transform = np.stack([mask_transform for i in range(get_transform.shape[1])], axis=1).astype('int')
        return self.ttu[:, :, 0, :] * full_mask_transform

    def construct_graph(self):
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        return fig, ax

    def plot_ttu(self, index, ax):
        xyz = [self.ttu[:, i, index, :].flatten() for i in range(0,3)]
        ax.scatter(*xyz)

    def plot_mti(self, exp, var, ax):
        mti = self.mask_time_interval(exp, var)
        xyz = [mti[:, i, :].flatten() for i in range(0, 3)]
        ax.scatter(*xyz)

