from scipy.signal import kaiserord, lfilter, firwin, medfilt, wiener
from scipy.interpolate import UnivariateSpline
import numpy


class lFilter:
    def __init__(self, z, dL, nyq, cutoff=0, width=0.1, ripple_db=120):
        self.z_max = z[-1]
        self.z = z
        self.dL = dL
        self.nyq = nyq
        self.width = width
        self.ripple_db = ripple_db
        if cutoff == 0:
            self.cutoff = self.nyq
        else:
            self.cutoff = cutoff

        self._dLz = UnivariateSpline(self.z, self.dL, k=3, s=0, ext=1)
        self._d2Lz = self._dLz.derivative()
        self.z = numpy.linspace(self.z[0], self.z[-1], len(self.z) * 10)
        # self.nyq = self.nyq * 10
        self.d2L = self._d2Lz(self.z)
        # self.d2L = medfilt(self._d2Lz(self.z), 51)
        # self.d2L = wiener(self._d2Lz(self.z))

        # self.N, self.beta = kaiserord(self.ripple_db, self.width)
        # self.delay = 0.5 * (self.N - 1) / (2 * self.nyq)
        # self.n_pad = numpy.int(numpy.ceil(self.delay / (self.z[1] - self.z[0])))
        # self.padding(self.n_pad)
        # self.taps = firwin(self.N, self.cutoff, window=('kaiser', self.beta), nyq=self.nyq)

        # self._d2Lz = UnivariateSpline(self.z - self.delay, self.raw_d2Lz(), k=2, s=0, ext=1)
        self._d2Lz = UnivariateSpline(self.z, self.d2L, k=2, s=0, ext=1)
        self._dLz = self._d2Lz.antiderivative()

    def padding(self, n):
        self.d2L = numpy.pad(self.d2L, (0, n), 'constant', constant_values=(0, 0))
        self.z = numpy.pad(self.z, (0, n), 'linear_ramp', end_values=(self.z[0], self.z[-1] + n * (self.z[1] - self.z[0])))

    def raw_d2Lz(self):
        return lfilter(self.taps, 1.0, self.d2L)

    def dLz(self):
        return self._dLz

    def d2Lz(self):
        return self._d2Lz
