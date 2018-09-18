from scipy.signal import kaiserord, lfilter, firwin
from scipy.interpolate import interp1d
from scipy.optimize import approx_fprime
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
            self.cutoff = self.nyq * 1 / 2
        else:
            self.cutoff = cutoff
        self.N, self.beta = kaiserord(self.ripple_db, self.width)
        self.delay = 0.5 * (self.N - 1) / (2 * self.nyq)
        self.n_pad = numpy.int(numpy.ceil(self.delay / (self.z[1] - self.z[0])))
        self.padding(self.n_pad)
        self.taps = firwin(self.N, self.cutoff, window=('kaiser', self.beta), nyq=self.nyq)
        self._dLz = interp1d(self.z - self.delay, self.raw_dLz(), kind="cubic", bounds_error=False, fill_value=0)

    def padding(self, n):
        self.dL = numpy.pad(self.dL, (0, n), 'constant', constant_values=(0, 0))
        self.z = numpy.pad(self.z, (0, n), 'linear_ramp', end_values=(self.z[0], self.z[-1] + n * (self.z[1] - self.z[0])))

    def raw_dLz(self):
        return lfilter(self.taps, 1.0, self.dL)

    def dLz(self):
        return self._dLz

    def d2Lz(self):
        eps = numpy.sqrt(numpy.finfo(float).eps)
        vf = numpy.vectorize(lambda z: 0 if z >= self.z_max - 2 * eps else approx_fprime([z], self._dLz, eps))

        def newfunc(*args, **kwargs):
            res = vf(*args, **kwargs)
            if len(res) == 1:
                return res[0]
            return res
        return newfunc
