import numpy as np
from class_observer import observer
from class_elipsoid import elipsiod

class homotheicTransform():
    def __init__(self, obs, el):
        self.el = el
        self.obs = obs
        self.ttu = self.tensor_time_unit()

    def y_hat(self):
        A = -1*(self.el.shape - self.obs.location)
        B = np.apply_along_axis(lambda x: x / np.linalg.norm(x), 1, A)
        return B

    def tensor_time_unit(self):
        A = []
        for i in np.linspace(0, self.el.dt*20, 20):
            A.append(np.stack([self.el.shape, self.y_hat(), self.el.shape + i * self.y_hat()]))
            self.el.translate_shape
            self.el.rotate_shape(np.array([1/np.sqrt(2), 0, 1/np.sqrt(2)]), np.pi/5)
        B = np.stack(A)
        C = np.einsum('ijkl -> klji', B)
        return C


if __name__ == '__main__':
    obs = observer(np.array([3, 1, 5]))
    el = elipsiod(np.array([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]),
                  0,
                  np.array([.55, .25, .1]),
                  np.array([1, 3, 2]),
                  np.array([0, 0, 0]),
                  .25)
    hT = homotheicTransform(obs, el)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    def construct_(ttu, index):
        xs = ttu[:,0,index,:].flatten()
        ys = ttu[:,1,index,:].flatten()
        zs = ttu[:,2,index,:].flatten()
        
        return xs, ys, zs

    xs0, ys0, zs0 = construct_(hT.ttu, 0)
    xs2, ys2, zs2 = construct_(hT.ttu, 2)
    xs3, ys3, zs3 = hT.obs.location[0], hT.obs.location[1], hT.obs.location[2]
    ax.scatter(xs2, ys2, zs2)
    ax.scatter(xs0, ys0, zs0)
    ax.scatter(xs3, ys3, zs3)
    
    plt.show()
