from scipy.interpolate import UnivariateSpline
from numpy import vectorize


class splinify:
    def __init__(self, z, l0, dL=None, d2L=None):
        self.z_max = z[-1]
        self.z = z
        self.dL = dL
        self.d2L = d2L
        self.l0 = l0
        if dL is not None:
            self._dLz = UnivariateSpline(self.z, self.dL, k=3, s=0, ext=1)
            self._d2Lz = self._dLz.derivative()
            self._Lz = self._dLz.antiderivative()
        elif d2L is not None:
            self._d2Lz = UnivariateSpline(self.z, self.d2L, k=3, s=0, ext=1)
            self._dLz = self._d2Lz.antiderivative()
            self._Lz = self._dLz.antiderivative()
        else:
            raise BaseException("No data to interpolate")

    def dLz(self):
        return vectorize(lambda x: -self._dLz(-x) if x > 0 else self._dLz(x))

    def d2Lz(self):
        return vectorize(lambda x: self._d2Lz(-x) if x > 0 else self._d2Lz(x))

    def Lz(self):
        return vectorize(lambda x: self._Lz(-x) + self.l0 if x > 0 else self._Lz(x) + self.l0)
