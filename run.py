from class_elipsoid import elipsiod
from class_observer import observer
from class_homotheicTransform import homotheicTransform
import numpy as np

if __name__ == '__main__':
    obs = observer(np.array([4, 2, 0]))
    el = elipsiod(np.array([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]),
                  0,
                  np.array([.55, .25, .1]),
                  np.array([8, 24, 16]),
                  np.array([.75, -.25, .015]),
                  .25)
    hT = homotheicTransform(obs, el, .5)

    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    hT.plot_ttu(0, ax)
    hT.plot_ttu(2, ax)

    plt.show()

