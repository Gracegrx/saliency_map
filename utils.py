import numpy as np

def makeGaussian(size, centre, fwhm=10):
    x = np.arange(0, size[1], 1, float)
    y = np.arange(0, size[0], 1, float)[:,np.newaxis]
    x0 = centre[0]
    y0 = centre[1]
    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

class Rect:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

    def isPointInRect(self, px, py):
        if px > self.x and px < (self.x + self.w):
            within_x = True
        else:
            within_x = False
        if py > self.y and py < (self.y + self.h):
            within_y = True
        else:
            within_y = False
        return within_x and within_y
