import numpy
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.signal import convolve
from signalProcessing import lFilter


class Convex_approx_1:
    def __init__(self, data_points, concave=False, c0=None, details=False):
        if concave:
            self.data_points = data_points * -1
        else:
            self.data_points = data_points
        self.n_points = len(self.data_points)
        self.c0 = c0
        self.distance = None
        self.distance_jac = None
        self.x0 = None
        self.eq_cons = None
        self.ineq_cons = None
        self.bounds = None
        self.details = details
        self.concave = concave
        if self.concave and self.c0 is not None:
            self.c0 *= -1

        self.build_distance()
        self.build_ineq_cons()
        self.build_eq_cons()
        self.build_bounds()
        self.build_x0()
        self.__precision = numpy.abs(numpy.min(data_points)) / 100000000
        if self.__precision == 0:
            self.__precision = numpy.abs(numpy.max(data_points)) / 100000000

        # print(self.__precision)

    def build_distance(self):
        def distance(x):
            res = 0
            for i in range(self.n_points):
                res += (x[i] - self.data_points[i])**2
            return res

        def distance_jac(x):
            res = []
            for i in range(self.n_points):
                res.append([2 * (x[i] - self.data_points[i])])
            return numpy.array(res)

        self.distance = distance
        self.distance_jac = distance_jac

    def build_ineq_cons(self):
        def fun(x):
            res = []
            for i in range(self.n_points - 2):
                res.append(x[i + 2] - 2 * x[i + 1] + x[i])
            if self.c0 is not None:
                res.append(x[1] - x[0] - self.c0)
            return numpy.array(res)

        jac_code = []
        for k in range(self.n_points - 2):
            temp_code = []
            for i in range(self.n_points):
                if i == k:
                    temp_code.append(1.0)
                elif i == k + 1:
                    temp_code.append(-2.0)
                elif i == k + 2:
                    temp_code.append(1.0)
                else:
                    temp_code.append(0)
            jac_code.append(temp_code)
        if self.c0 is not None:
            temp = numpy.zeros(self.n_points)
            temp[1] = 1
            temp[0] = -1
            jac_code.append(temp)

        ineq_cons = {
            'type': 'ineq',
            'fun': fun,
            'jac': lambda x: numpy.array(jac_code)
        }

        self.ineq_cons = ineq_cons

    def build_eq_cons(self):
        first_jac = []
        second_jac = []

        for i in range(self.n_points):
            if i == 0:
                first_jac.append(1)
                second_jac.append(0)
            elif i == self.n_points - 1:
                first_jac.append(0)
                second_jac.append(1)
            else:
                first_jac.append(0)
                second_jac.append(0)

        eq_cons = {
            'type': 'eq',
            'fun': lambda x: numpy.array([x[0] - self.data_points[0], x[-1] - self.data_points[-1]]),
            'jac': lambda x: numpy.array([first_jac, second_jac])
        }

        self.eq_cons = eq_cons

    def build_bounds(self):
        self.bounds = Bounds(numpy.ones(self.n_points) * numpy.min(self.data_points), numpy.ones(self.n_points) * numpy.max(self.data_points))

    def build_x0(self):
        x0 = numpy.zeros(self.n_points)
        y0 = numpy.zeros(self.n_points)
        x0[0] = self.data_points[0]
        for i in range(self.n_points - 1):
            min_der = numpy.infty
            for k in range(i + 1, self.n_points):
                min_der = min(min_der, (self.data_points[k] - x0[i]) / (k - i))
            y0[i] = min_der
            x0[i + 1] = x0[i] + min_der
        self.x0 = x0

    def minimize(self):
        res = minimize(self.distance, self.x0, jac=self.distance_jac, method='SLSQP', constraints=[self.eq_cons, self.ineq_cons], options={'ftol': self.__precision, 'disp': self.details, 'maxiter': 1000}, bounds=self.bounds)
        res = res.x
        if self.concave:
            res = res * -1
        return (res, res[-1] - res[-2])


class Convex_approx:
    def __init__(self, dLz_z, dLz, details=False, order=1):
        self.dLz_z = dLz_z
        self.dLz = dLz
        self.order = order
        self.dLz[0] = 0  # evite des effets de bord
        self.dLz[-1] = 0  # evite des effets de bord
        self._d2Lz = self.discrete_fprime(self.dLz, self.dLz_z)
        self._d2Lz[0] = 0  # evite des effets de bord
        self._d2Lz[-1] = 0  # evite des effets de bord
        self._d3Lz = self.discrete_fprime(self._d2Lz, self.dLz_z)
        self._d4Lz = self.discrete_fprime(self._d3Lz, self.dLz_z)
        self.details = details

    def discrete_fprime(self, f, z):
        pas = z[1] - z[0]
        f1 = numpy.roll(f, -1)
        f0 = numpy.roll(f, 1)
        return (f1 - f0) / (2 * pas)

    def _guess_sign_change(self, data, window=1, hysteresis=0):
        data = self.moving_average(data, window)
        res = []
        positive = True
        for i in range(len(data)):
            if positive and data[i] < -hysteresis:
                positive = False
                res.append(i)
            elif not positive and data[i] > hysteresis:
                positive = True
                res.append(i)
        return res

    def guess_sign_change(self, data):
        window = 1
        res = numpy.zeros(7)
        hysteresis = numpy.max(data) / 10
        while len(res) > 1 + self.order:
            res = self._guess_sign_change(data, window, hysteresis)
            window += 1
        return res

    def moving_average(self, data, window):
        weights = numpy.repeat(1.0, window) / window
        sma = convolve(data, weights, 'same')
        return sma

    def run_approx(self):
        n = len(self._d3Lz)
        if n % 2 == 0:
            n = n // 2
        else:
            n = n // 2 + 1
        if self.order == 1:
            data = self.dLz[:n]
        else:
            data = self._d2Lz[:n]
        if self.order == 1:
            res = self.guess_sign_change(self._d3Lz[:n])
        else:
            res = self.guess_sign_change(self._d4Lz[:n])
        split_data = []
        n = len(res)
        for i in range(n):
            if i == 0:
                split_data.append(data[0:res[i]])
            else:
                split_data.append(data[(res[i - 1] - 1):res[i]])
        split_data.append(data[(res[-1] - 1):])
        concave = False
        final_data = []
        # c0 = None
        for i in range(len(split_data)):
            temp = Convex_approx_1(split_data[i], concave, details=self.details)
            res = temp.minimize()
            # final_data.append(res[0])
            if i == len(split_data) - 1:
                final_data = numpy.concatenate((final_data, res[0]))
            else:
                final_data = numpy.concatenate((final_data, res[0][:-1]))
            # c0 = res[1]
            concave = not concave
        temp = []
        for i in range(1, len(final_data)):
            if self.order == 1:
                temp.append(-final_data[-i - 1])
            else:
                temp.append(final_data[-i - 1])
        final_data = numpy.concatenate((final_data, temp))
        return final_data
