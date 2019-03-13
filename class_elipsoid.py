import numpy as np

class elipsiod():
    def __init__(self, unit, theta, magnitudes, init_location, velocity, dt):
        self.init_location = init_location
        self.velocity = velocity
        self.basis = self.construct_basis(unit, theta, magnitudes)
        self.shape = self.construct_shape()
        self.dt = dt
        self.fig, self.ax = self.construct_graph()

    def construct_basis(self, unit, theta, magnitudes):
        a11 = np.cos(theta) + (unit[0]**2)*(1 - np.cos(theta))
        a12 = unit[0]*unit[1]*(1 - np.cos(theta)) - unit[2]*np.sin(theta)
        a13 = unit[0]*unit[2]*(1 - np.cos(theta)) + unit[1]*np.sin(theta)
        a21 = unit[0]*unit[1]*(1 - np.cos(theta)) + unit[2]*np.sin(theta)
        a22 = np.cos(theta) + (unit[1]**2)*(1 - np.cos(theta))
        a23 = unit[1]*unit[2]*(1 - np.cos(theta)) - unit[0]*np.sin(theta)
        a31 = unit[0]*unit[2]*(1 - np.cos(theta)) - unit[1]*np.sin(theta)
        a32 = unit[1]*unit[2]*(1 - np.cos(theta)) + unit[0]*np.sin(theta)
        a33 = np.cos(theta) + (unit[2]**2)*(1 - np.cos(theta))

        return magnitudes * np.array([[a11, a12, a13], [a21, a22, a23], [a31, a32, a33]])

    def construct_shape(self):
        domain_sample = 2*np.pi*np.random.random_sample(size=(50, 2))
        sphere_sample = np.apply_along_axis(self.make_sphere, 1, domain_sample)
        range_sample = sphere_sample @ self.basis.T
        return range_sample + self.init_location

    def make_sphere(self, theta):
        return np.array([np.cos(theta[0])*np.sin(theta[1]),
                         np.sin(theta[0])*np.sin(theta[1]),
                                          np.cos(theta[1])])

    def translate_shape(self):
        self.init_location = self.init_location + self.dt*self.velocity
        self.shape += self.dt*self.velocity

    def rotate_shape(self, unit, theta):
        rotation_m = self.construct_basis(unit, theta*self.dt, np.array([1, 1, 1]))
        self.shape = self.shape - self.init_location
        self.shape = self.shape @ rotation_m
        self.shape = self.shape + self.init_location

    def construct_graph(self):
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        return fig, ax

    def plot(self, show=True):
        from mpl_toolkits.mplot3d import Axes3D
        import matplotlib.pyplot as plt

        fig = self.fig
        ax = self.ax

        ax.scatter(self.shape[:, 0], self.shape[:, 1], self.shape[:, 2], color='blue')

        if show:
            plt.show()
