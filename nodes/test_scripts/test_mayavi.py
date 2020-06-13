import time
import numpy as np
from mayavi import mlab

f = mlab.figure()
V = np.random.randn(20, 20, 20)
s = mlab.contour3d(V, contours=[0])

@mlab.animate(delay=10)
def anim():
    i = 0
    while i < 5:
        time.sleep(1)
        s.mlab_source.set(scalars=np.random.randn(20, 20, 20))
        i += 1
        yield

anim()
mlab.show()