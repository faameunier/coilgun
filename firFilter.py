from scipy.signal import kaiserord, lfilter, firwin
import matplotlib.pyplot as plt
import pltHelper
import numpy
import config


class lFilter:
    def __init__(self, z, fun, sampling_freq, cutoff=0, width=0.1, ripple_db=120):
        self._z = z
        self._fun = fun
        self.nyq = sampling_freq
        self.width = width
        self.ripple_db = ripple_db
        if cutoff == 0:
            self.cutoff = self.nyq / (2 * config.OVERSAMPLING)
        else:
            self.cutoff = cutoff

        self.z = numpy.linspace(self._z[0], self._z[-1], len(self._z) * 10)

        self.N, self.beta = kaiserord(self.ripple_db, self.width)
        if self.N % 2 == 0:
            self.N += 1
        self.delay = numpy.int((self.N - 1) / 2)
        self.n_pad = numpy.int(numpy.ceil(self.delay / (self.z[1] - self.z[0])))
        # print(self.n_pad)
        self.padding(self.delay)
        # print(self.fun)
        self.taps = firwin(self.N, self.cutoff, window=('kaiser', self.beta), nyq=self.nyq)

        self.fun = lfilter(self.taps, 1.0, self.fun)

        # print("fun", self.fun)
        # print("z", self.z)

    def padding(self, n):
        self.fun = numpy.pad(self._fun, (0, n), 'constant', constant_values=(0, 0))
        self.z = numpy.pad(self.z, (0, n), 'linear_ramp', end_values=(self.z[0], self.z[-1] + n * (self.z[1] - self.z[0])))

    def output(self):
        return self.fun[self.delay:]

    def plot(self):
        plt.plot(self._z, self._fun, color=(1, 0, 0), label="input")
        plt.plot(self._z, self.fun[self.delay:], color=(0, 0, 1), label="FIR filtered")
        plt.xlabel(r"$z(m)$")
        plt.ylabel(r"$H.m^{-2}$")
        plt.legend()
        plt.show()
