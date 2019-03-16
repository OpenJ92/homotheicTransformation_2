from class_elipsoid import elipsiod
from class_observer import observer
from class_homotheicTransform import homotheicTransform
import numpy as np

if __name__ == '__main__':
    obs = observer(np.array([10, 10, 10]))
    obs2 = observer(np.array([30, 30, 30]))
    obs3 = observer(np.array([40, 50, 0]))
    el = elipsiod(np.array([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]),
                  0,
                  np.array([.55, .25, .1]),
                  np.array([0, 0, 0]),
                  np.array([.75, -.25, .015]),
                  .25)
    el2 = elipsiod(np.array([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]),
                  0,
                  np.array([.55, .25, .1]),
                  np.array([0, 0, 0]),
                  np.array([.75, -.25, .015]),
                  .25)
    el3 = elipsiod(np.array([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]),
                  0,
                  np.array([.55, .25, .1]),
                  np.array([0, 0, 0]),
                  np.array([.75, -.25, .015]),
                  .25)
    time_intervals = 200
    hT = homotheicTransform(obs, el, 1.5*np.linalg.norm(el.velocity), time_intervals)
    hT2 = homotheicTransform(obs2, el2, 1.5*np.linalg.norm(el2.velocity), time_intervals)
    hT3 = homotheicTransform(obs3, el3, 1.5*np.linalg.norm(el3.velocity), time_intervals)
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    hT.plot_ttu(0, ax)
    hT2.plot_ttu(2, ax)
    hT3.plot_ttu(2, ax)
    #hT.plot_mti(5, .01, ax)
    #hT2.plot_mti(5, .01, ax)
    #hT3.plot_mti(5, .01, ax)
    ax.scatter(*obs.location)
    ax.scatter(*obs2.location)
    ax.scatter(*obs3.location)
    plt.show()
