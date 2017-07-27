import numpy as np
from utils import makeGaussian
import matplotlib.pyplot as plt

class Fixation:
    def __init__(self, x, y, start, dur, loc_x, loc_y, size_x, size_y):
        self.x = x
        self.y = y
        self.start = start
        self.dur = dur
        self.loc_x = loc_x
        self.loc_y = loc_y
        self.size_x = size_x
        self.size_y = size_y


    def __repr__(self):
        return "fix @({x}, {y}), t={start}, d={dur}".format(
            x=self.x, y=self.y, start=self.start, dur=self.dur)

    def __str__(self):
        return "fix @({x}, {y}), t={start}, d={dur}".format(
            x=self.x, y=self.y, start=self.start, dur=self.dur)

    def convert_to_saliency_map(self, size):
        x = int((self.x-self.loc_x)/self.size_x * (size[0] - 1))
        y = int((self.y-self.loc_y)/self.size_y * (size[1] - 1))
        gaussian = makeGaussian(size=size, centre=(x, y))
        #print gaussian
        #print "center = ", x, y
        return np.expand_dims(gaussian, 2)
        #return gaussian

class FixationsList:
    def __init__(self, fix_list):
        self.fixations = fix_list

    @classmethod
    def from_pos(cls, x_pos, y_pos, start, dur, loc_x, loc_y, size_x, size_y):
        fixations = []
        for x, y, s, d, lx, ly, sx, sy in zip(x_pos, y_pos, start, dur, loc_x, loc_y, size_x, size_y):
            fix = Fixation(x, y, s, d, lx, ly, sx, sy)
            fixations.append(fix)
        return cls(fixations)

    def __getitem__(self, i):
        return self.fixations[0]

    def __repr__(self):
        return str(self.fixations)

    def __add__(self, b):
        fixations = self.fixations + b.fixations
        return FixationsList(fixations)

    def __len__(self):
        return len(self.fixations)

    def sort(self):
        self.fixations = sorted(self.fixations, key=lambda x: x.start)

    def __iter__(self):
        for x in self.fixations:
            yield x