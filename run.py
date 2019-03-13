from class_elipsoid import elipsiod
from class_observer import observer
from class_homotheicTransform import homotheicTransform
import numpy as np

obs = observer(np.array([3, 1, 5]))
el = elipsiod(np.array([1/np.sqrt(3), 1/np.sqrt(3), 1/np.sqrt(3)]),
              0,
              np.array([1, .25, 2]),
              np.array([1, 3, 2]),
              np.array([-.5, -.5, -.5]),
              .25)
hT = homotheicTransform(obs, el)
