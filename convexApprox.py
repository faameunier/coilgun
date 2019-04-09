import numpy
from scipy.optimize import minimize
from scipy.optimize import Bounds
from scipy.signal import convolve
import matplotlib.pyplot as plt
from firFilter import lFilter
import config


class Convex_approx_1:
    def __init__(self, data_points, concave=False, c0=None, details=False):
        """Best convex approximation of a dataset

        Find the best convex (or concave) approximations of a given dataset.
        This class creates new data points that will minimize the loss
        between the input data and the new one while still forcing convexity
        (or concavity).

        It is possible to enforce the initial derivative via c0.

        Arguments:
            data_points {list} -- initial datapoints, regurlaly sampled

        Keyword Arguments:
            concave {bool} -- switch to a concave approximation (default: {False})
            c0 {float} -- Initial local derivative (default: {None})
            details {bool} -- verbose optimization or not (default: {False})
        """
        if concave:
            self.data_points = numpy.array(data_points) * -1
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

        if self.n_points > 2:
            self.build_distance()
            self.build_ineq_cons()
            self.build_eq_cons()
            self.build_bounds()
            self.build_x0()

        # print(self.__precision)

    def build_distance(self):
        """Builds distance and jacobian function

        Based on the input dataset builds two function
        to compute the error and jacobian of the error.
        """
        def distance(x):
            error = x - self.data_points
            return numpy.sum(error**2)

        def distance_jac(x):
            error = x - self.data_points
            return 2 * error

        self.distance = distance
        self.distance_jac = distance_jac

    def build_ineq_cons(self):
        """Build convexe (or concave) inequalities

        Given the size of the dataset builds all
        inequalities to enforce convexity or concavity
        of the output data points.

        Takes into account the initial derivative if specified
        via c0 during instanciation.
        """
        def fun(x):
            x_1 = numpy.roll(x, 1)
            x_2 = numpy.roll(x, 2)
            conv_cons = (x - 2 * x_1 + x_2)[2:]
            if self.c0 is not None:
                return numpy.append(conv_cons, x[1] - x[0] - self.c0)
            else:
                return conv_cons

        jac_code = []
        base_derivative = numpy.zeros(self.n_points)
        base_derivative[2] = 1
        base_derivative[1] = -2
        base_derivative[0] = 1
        for k in range(self.n_points - 2):
            jac_code.append(numpy.roll(base_derivative, k))
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
        """Builds constraints to force the bounds of the output datapoints

        The two extreme bounds of the output data should match the input data.
        This function suppose that first bound has a nul derivative while the
        second has a nul second order derivative.

        This is a large assumption and a future version could allow some
        personnalisation. However it is useful if the sign of the second order
        derivative changes in time.
        """
        first = numpy.zeros(self.n_points)
        first[0] = 1
        second = numpy.zeros(self.n_points)
        second[-1] = 1

        eq_cons = {
            'type': 'eq',
            'fun': lambda x: numpy.array([x[0] - self.data_points[0], x[-1] - self.data_points[-1]]),
            'jac': lambda x: numpy.array([first, second])
        }

        self.eq_cons = eq_cons

    def build_bounds(self):
        """Set some bounds for the output data.

        All datapoints should remain between the min and max
        of all inputs. This is just to speed up the optimization.
        """
        self.bounds = Bounds(numpy.min(self.data_points), numpy.max(self.data_points))

    def build_x0(self):
        """Initial output guess

        As the output's second derivative should keep
        constant sign, we can set a first guess of the output
        being the convex hull.
        """
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
        """Performs the actual filter

        Perform the filtering by solving
        a minimization problem.

        Return:
            np.array -- the filtered dataset
        """
        cons = []
        if self.n_points > 2:  # if there are only 2 points, there are no ineq cons
            cons = [self.eq_cons, self.ineq_cons]
            res = minimize(self.distance, self.x0, jac=self.distance_jac, method='SLSQP', constraints=cons, options={'ftol': config.CONVEX_TOL, 'disp': self.details, 'maxiter': config.CONVEX_MAX_ITER}, bounds=self.bounds)
            res = res.x
            if self.concave:
                res = res * -1
            return (res, res[-1] - res[-2])
        else:
            c = None
            if self.n_points == 2:
                c = self.data_points[1] - self.data_points[0]
            return (self.data_points, c)


class Convex_approx_v2:
    def __init__(self, data_points, indexes, details=False):
        """Best convex approximation of a dataset

        Find the best convex/concave constant piecewise approximation
        of the data where the sign reversals accures at indexes.

        Supposes that f''(0) >= 0

        Arguments:
            data_points {list} -- initial datapoints, regurlaly sampled
            indexes {list} -- list of second order derivative sign reversals

        Keyword Arguments:
            details {bool} -- verbose optimization or not (default: {False})
        """
        self.data_points = data_points
        self.n_points = len(self.data_points)
        self.indexes = list(indexes)
        self.distance = None
        self.distance_jac = None
        self.x0 = None
        self.eq_cons = None
        self.ineq_cons = None
        self.bounds = None
        self.details = details

        if self.n_points > 2:
            self.preprocess_indexes()
            self.build_distance()
            self.build_ineq_cons()
            self.build_eq_cons()
            self.build_bounds()
            self.build_x0()

        # print(self.__precision)

    def preprocess_indexes(self):
        mask = numpy.zeros_like(self.data_points, dtype=bool)
        indexes = self.indexes
        convex = False
        if indexes[0] == 0:
            convex = True
        else:
            indexes.insert(0, 0)
        if indexes[-1] != self.n_points - 1:
            indexes.append(self.n_points - 1)
        for k in range(self.n_points):
            if k in indexes:
                convex = not convex
            if convex:
                mask[k] = True
            else:
                mask[k] = False
        self.convex_mask = mask
        self.concave_mask = numpy.logical_not(mask)
        for i in indexes:
            self.convex_mask[i] = False
            self.concave_mask[i] = False

    def build_distance(self):
        """Builds distance and jacobian function

        Based on the input dataset builds two function
        to compute the error and jacobian of the error.
        """
        def distance(x):
            error = x - self.data_points
            return numpy.sum(error**2)

        def distance_jac(x):
            error = x - self.data_points
            return 2 * error

        self.distance = distance
        self.distance_jac = distance_jac

    def build_ineq_cons(self):
        """Build convexe (or concave) inequalities

        Given the size of the dataset builds all
        inequalities to enforce convexity or concavity
        of the output data points.

        Takes into account the initial derivative if specified
        via c0 during instanciation.
        """
        def fun(x):
            x_1 = numpy.roll(x, 1)  # x_{-1}
            x_2 = numpy.roll(x, -1)  # x_{+1}
            ineq_cons = (x_1 - 2 * x + x_2)
            conv_ma = ineq_cons[self.convex_mask]
            conc_ma = -ineq_cons[self.concave_mask]
            res = numpy.concatenate((conv_ma, conc_ma))
            return res

        jac_code = []
        jac_code_conv = []
        jac_code_conc = []
        base_derivative = numpy.zeros(self.n_points)
        base_derivative[2] = 1
        base_derivative[1] = -2
        base_derivative[0] = 1
        convex = True
        for k in range(self.n_points):
            if self.convex_mask[k] == True:
                jac_code_conv.append(numpy.roll(base_derivative, k - 1))
            elif self.concave_mask[k] == True:
                jac_code_conc.append(-1 * numpy.roll(base_derivative, k - 1))
        jac_code = jac_code_conv + jac_code_conc

        plt.plot(jac_code)
        plt.show()

        print(numpy.array(jac_code).shape)
        ineq_cons = {
            'type': 'ineq',
            'fun': fun,
            'jac': lambda x: numpy.array(jac_code)
        }

        self.ineq_cons = ineq_cons

    def build_eq_cons(self):
        """Builds constraints to force the bounds of the output datapoints

        The two extreme bounds of the output data should match the input data.
        This function suppose that first bound has a nul derivative while the
        second has a nul second order derivative.

        This is a large assumption and a future version could allow some
        personnalisation. However it is useful if the sign of the second order
        derivative changes in time.
        """
        def fun(x):
            temp = []
            for i in self.indexes[1:-1]:
                temp.append(x[i - 1] - 2 * x[i] + x[i + 1])
            return temp

        jac_code = []
        base_derivative = numpy.zeros(self.n_points)
        base_derivative[2] = 1
        base_derivative[1] = -2
        base_derivative[0] = 1
        for i in self.indexes[1:-1]:
            jac_code.append(numpy.roll(base_derivative, i - 1))

        # borders
        first = numpy.zeros(self.n_points)
        first[0] = 1
        second = numpy.zeros(self.n_points)
        second[-1] = 1

        eq_cons = {
            'type': 'eq',
            'fun': lambda x: numpy.array([x[0] - self.data_points[0], x[-1] - self.data_points[-1]] + fun(x)),
            'jac': lambda x: numpy.array([first, second] + jac_code)
        }

        self.eq_cons = eq_cons

    def build_bounds(self):
        """Set some bounds for the output data.

        All datapoints should remain between the min and max
        of all inputs. This is just to speed up the optimization.
        """
        self.bounds = Bounds(numpy.min(self.data_points), numpy.max(self.data_points))

    def build_x0(self):
        """Initial output guess

        As the output's second derivative should keep
        constant sign, we can set a first guess of the output
        being the convex hull.
        """
        def sub_optimal(data_points):
            n_points = len(data_points)
            x0 = numpy.zeros(n_points)
            y0 = numpy.zeros(n_points)
            x0[0] = data_points[0]
            for i in range(n_points - 1):
                min_der = numpy.infty
                for k in range(i + 1, n_points):
                    min_der = min(min_der, (data_points[k] - x0[i]) / (k - i))
                y0[i] = min_der
                x0[i + 1] = x0[i] + min_der
            return x0
        res = []
        convex = False
        for k in range(len(self.indexes) - 1):
            convex = not convex
            interval = (self.indexes[k], self.indexes[k + 1] + 1)
            temp = self.data_points[interval[0]:interval[1]]
            if convex:
                res.append(sub_optimal(temp)[:-1])
            else:
                res.append(-1 * sub_optimal(-1 * temp)[:-1])
        res = numpy.concatenate(res)
        res = numpy.append(res, self.data_points[-1])
        self.x0 = res

    def minimize(self):
        """Performs the actual filter

        Perform the filtering by solving
        a minimization problem.

        Return:
            np.array -- the filtered dataset
        """
        cons = []
        # plt.plot(self.data_points)
        # plt.plot(self.x0)
        # plt.show()
        if self.n_points > 2:
            cons = [self.eq_cons, self.ineq_cons]
            res = minimize(self.distance, self.x0, jac=self.distance_jac, method='SLSQP', constraints=cons, options={'ftol': config.CONVEX_TOL, 'disp': self.details, 'maxiter': config.CONVEX_MAX_ITER}, bounds=self.bounds)
            res = res.x
            return res
        else:
            return self.data_points


class Convex_approx:
    def __init__(self, dLz_z, dLz, est_freq, details=False):
        self.dLz_z = dLz_z
        self.dLz = dLz
        self.order = 2
        self.dLz[0] = 0  # evite des effets de bord
        self.dLz[-1] = 0  # evite des effets de bord
        self._d2Lz = self.discrete_fprime(self.dLz, self.dLz_z)
        self._d2Lz[0] = 0  # evite des effets de bord
        self._d2Lz[-1] = 0  # evite des effets de bord
        self.est_freq = est_freq
        self.details = details

    def discrete_fprime(self, f, z):
        """Compute the derivative of a function

        Arguments:
            f {np.array} -- input data
            z {np.array} -- sampling points
        """
        pas = z[1] - z[0]
        f1 = numpy.roll(f, -1)
        f0 = numpy.roll(f, 1)
        res = (f1 - f0) / (2 * pas)
        res[0] = res[1]
        res[-1] = res[-2]
        return res

    def _guess_sign_change(self, data, window=1, hysteresis=0):
        """Guess where the second order derivative sign change

        This is an heuristic and could be improved. It compute the average value of
        the data on a given window and compare it to a hysteresis value.
        If the average is beyond the hysteresis it is considered as a sign inversion.

        Arguments:
            data {np.array} -- input data, the second order derivative of the initial dataset

        Keyword Arguments:
            window {number} -- size of the window (default: {1})

        Returns:
            list -- list of indexes where the sign changes
        """
        data = self.moving_average(data, window)
        res = []
        positive = True
        # plt.plot(data)
        # plt.show()
        hysteresis = numpy.max(numpy.abs(data)) / 10
        for i in range(len(data)):
            if positive and data[i] < -hysteresis:
                positive = False
                res.append(i)
            elif not positive and data[i] > hysteresis:
                positive = True
                res.append(i)
        return res

    def guess_sign_change(self, data):
        """Helper to guess sign change

        Recursively calls _guess_sign_change
        with an increasing window and some
        reasonable hysteresis until the number
        number of inversion matches the order
        of the filter.

        Arguments:
            data {np.array} -- input data, the second order derivative of the initial dataset

        Returns:
            tuple -- list of indexes where the sign changes and last window used
        """
        window = 1
        # plt.plot(data)
        # plt.show()
        if self.order < 0:
            raise Exception("Failed to guess splitting points")
        res = numpy.zeros(2 + self.order)
        while len(res) > self.order + 1:
            res = self._guess_sign_change(data, window)
            window += 1
            if window == len(data):
                self.order -= 1
                res = self.guess_sign_change(data)
                break
        # print(res)
        return res

    def moving_average(self, data, window):
        """centered moving average"""
        weights = numpy.repeat(1.0, window) / window
        sma = convolve(data, weights, 'valid')
        return sma

    def run_approx(self):
        n = len(self._d2Lz)
        if n % 2 == 0:
            n = n // 2
        else:
            n = n // 2 + 1
        data = numpy.copy(self._d2Lz[:n])
        data = lFilter(self.dLz_z[:n], data, sampling_freq=2 * config.OVERSAMPLING * self.est_freq)
        # data.plot()
        data = data.output()
        indexes = self.guess_sign_change(self.discrete_fprime(self.discrete_fprime(data, self.dLz_z), self.dLz_z))
        # print(indexes)
        data = self._d2Lz[:n]
        if config.CONVEX_V2:
            final_data =  Convex_approx_v2(data, indexes, True).minimize()
        # plt.plot(self.dLz_z[:n], data, c=(1, 0, 0))
        # for ix in indexes:
        #     plt.axvline(self.dLz_z[ix], c='k')
        # plt.xlabel(r"$z(m)$")
        # plt.ylabel(r"$H.m^{-2}$")
        # plt.show()
        else:
            split_data = []
            n = len(indexes)
            for i in range(n):
                if i == 0:
                    split_data.append(data[0:indexes[i]])
                else:
                    split_data.append(data[(indexes[i - 1] - 1):indexes[i]])
            split_data.append(data[(indexes[-1] - 1):])
            concave = False
            final_data = []
            # c0 = None
            for i in range(len(split_data)):
                temp = Convex_approx_1(split_data[i], concave, details=self.details)  # , c0=c0)
                res = temp.minimize()
                # final_data.append(res[0])
                if i == len(split_data) - 1:
                    final_data = numpy.concatenate((final_data, res[0]))
                else:
                    final_data = numpy.concatenate((final_data, res[0][:-1]))
                # c0 = res[1]
                concave = not concave
        # plt.plot(data)
        # plt.plot(final_data)
        # plt.show()
        temp = []
        for i in range(1, len(final_data)):
            temp.append(final_data[-i - 1])
        final_data = numpy.concatenate((final_data, temp))
        return final_data
