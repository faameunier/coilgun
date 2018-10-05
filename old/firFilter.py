from scipy.signal import kaiserord, lfilter, firwin
import matplotlib.pyplot as plt
import pltHelper
import numpy


class lFilter:
    def __init__(self, z, fun, nyq, cutoff=0, width=0.1, ripple_db=120):
        self._z = z
        self._fun = fun
        self.nyq = nyq
        self.width = width
        self.ripple_db = ripple_db
        if cutoff == 0:
            self.cutoff = 1 / len(self._z)
        else:
            self.cutoff = cutoff

        self.z = numpy.linspace(self._z[0], self._z[-1], len(self._z) * 10)

        self.N, self.beta = kaiserord(self.ripple_db, self.width)
        if self.N % 2 == 0:
            self.N += 1
        self.delay = numpy.int((self.N - 1) / 2)
        self.n_pad = numpy.int(numpy.ceil(self.delay / (self.z[1] - self.z[0])))
        print(self.n_pad)
        self.padding(self.delay)
        print(self.fun)
        self.taps = firwin(self.N, self.cutoff, window=('kaiser', self.beta), nyq=self.nyq, pass_zero=False)

        self.fun = lfilter(self.taps, 1.0, self.fun)

        print("fun", self.fun)
        print("z", self.z)

    def padding(self, n):
        self.fun = numpy.pad(self._fun, (0, n), 'constant', constant_values=(0, 0))
        self.z = numpy.pad(self.z, (0, n), 'linear_ramp', end_values=(self.z[0], self.z[-1] + n * (self.z[1] - self.z[0])))

    def ouput(self):
        return self.fun

    def plot(self):
        plt.plot(self._fun, color=(1, 0, 0))
        plt.plot(self.fun[self.delay:], color=(0, 0, 1))
        plt.show()
