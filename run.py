from class_elipsoid import elipsiod
from class_observer import observer
from class_homotheicTransform import homotheicTransform
import numpy as np

if __name__ == '__main__':
    # look to make these plot functions methods of the class
    obs = observer(np.array([0, 0, 0]))
    el = elipsiod(np.array([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]),
                  0,
                  np.array([.55, .25, .1]),
                  np.array([1, 3, 2]),
                  np.array([40, 0, 0]),
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

    def plot_transformed_obj(mti):
        K = [mti[:,i,:].flatten() for i in range(mti.shape[1])]
        ax.scatter(*K)
        plt.show()
