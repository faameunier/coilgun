from scipy.integrate import odeint
import numpy
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pltHelper
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


class gaussSolver:
    __n = 10000

    def __init__(self, l, C, E, m, R):
        self.l_splin = l
        self.C = C
        self.R = R
        self.E = E
        self.m = m
        lambd = self.R / (2 * self.l_splin.l0)
        omega_2 = 1 / (self.l_splin.l0 * self.C)
        if lambd**2 <= omega_2:
            # print("osc")
            self._t_max = 10 * (1 / lambd)
        elif lambd**2 > omega_2:
            # print("amort")
            self._t_max = 10 * (1 / (lambd - numpy.sqrt(lambd**2 - omega_2)))
        self._t_max = max(self._t_max, 0.050)
        self.t = numpy.linspace(0, self._t_max, self.__n)

    def i_eq(self, y, t):
        i, i_p, z, z_p = y
        return [
            i_p,
            -1 / (self.l_splin.Lz()(z) * self.C) * ((self.R + 2 * z_p * self.l_splin.dLz()(z)) * self.C * i_p + (1 + self.C * (i**2 / self.m / 2 * self.l_splin.dLz()(z)**2 + z_p**2 * self.l_splin.d2Lz()(z))) * i),
            z_p,
            i**2 / self.m / 2 * self.l_splin.dLz()(z),
        ]

    def y0(self, z0, z_p0):
        return [
            0,
            -self.E / self.l_splin.Lz()(z0),
            z0,
            z_p0,
        ]

    def solve(self, z0, z0_p):
        return odeint(self.i_eq, self.y0(z0, z0_p), self.t)

    def plot_single(self, result):
        # print(result[:])
        ax1 = plt.subplot(211)
        plt.plot(self.t, result[:, 0], color=(0, 0, 1), label="i(t)")
        plt.setp(ax1.get_xticklabels())

        plt.subplot(212, sharex=ax1)
        plt.plot(self.t, result[:, 3], color=(0, 0, 1), label="dz/dt(t)")

        plt.show()

    def plot_multiple(self, results):
        n = len(results)

        ax1 = plt.subplot(211)
        for i in range(n):
            plt.plot(self.t, results[i][:, 0], color=(i / n, 0, 1 - i / n), label="i(t) - " + str(i))
        plt.setp(ax1.get_xticklabels())

        plt.subplot(212, sharex=ax1)
        for i in range(n):
            plt.plot(self.t, results[i][:, 3], color=(i / n, 0, 1 - i / n), label="dz/dt(t) - " + str(i))

        plt.show()

    def computeMaxEc(self, result):
        v_max = result[:, 3][-1]
        print(v_max)
        return 1 / 2 * self.m * v_max**2

    def computeMaxE(self, result):
        return 1 / 2 * self.C * self.E**2

    def computeTau(self, result):
        return self.computeMaxEc(result) / self.computeMaxE(result)

    def _dicho_opt(self, bound, ite=50):
        temp_tau = 0
        i = 1
        res = (self.solve(bound, 0), self.solve(0, 0))
        z0 = (bound, 0)
        while i < ite:
            temp_tau = (self.computeTau(res[0]), self.computeTau(res[1]))
            print(z0, temp_tau)
            if temp_tau[0] > temp_tau[1]:
                z0 = (z0[0], numpy.mean(z0))
                res = (res[0], self.solve(z0[1], 0))
            else:
                z0 = (numpy.mean(z0), z0[1])
                res = (self.solve(z0[0], 0), res[1])
            i += 1
        if temp_tau[0] > temp_tau[1]:
            return (z0[0], res[0])
        else:
            return (z0[1], res[1])

    def _linear_opt(self, bound, epsilon=0.0005, plot=False, plot3d=False):
        res = []
        z0 = []
        i = 0

        n = int(numpy.abs(bound / epsilon))
        for i in range(n):
            print(i / n * 100)
            z0.append(bound + epsilon * i)
            res.append(self.solve(bound + epsilon * i, 0))
        res = numpy.array(res)

        if plot:
            colors = []
            ax1 = plt.subplot(211)
            for i in range(n):
                colors.append((i / n, 0, 1 - i / n))
                plt.plot(self.t, res[i][:, 3], color=(i / n, 0, 1 - i / n), label="dz/dt(t) - " + str(i))
            plt.setp(ax1.get_xticklabels())

            ax2 = plt.subplot(212)
            line = []
            for k in range(len(z0)):
                line.append([z0[k], res[k][:, 3][-1]])
            line = numpy.array(line).reshape(-1, 1, 2)
            line = numpy.hstack([line[:-1], line[1:]])
            coll = LineCollection(line, colors=colors)
            ax2.add_collection(coll)
            ax2.autoscale_view()
            plt.show()

        if plot3d:
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            X, Y = numpy.meshgrid(self.t, z0)
            # print(numpy.shape(X))
            # print(numpy.shape(Y))
            # print(numpy.shape(res))
            ax.plot_surface(X, Y, res[:, :, 3], cmap=cm.viridis, linewidth=0, antialiased=False, rcount=100, ccount=100)
            plt.show()

        print(res[:, :, 3])
        arg = numpy.argmax(numpy.array(res[:, :, 3])[:, -1])
        return (z0[arg], res[arg])

    def computeOptimal(self, bound, method="linear"):
        if method not in ["dicho", "linear"]:
            raise BaseException("Only linear opt available.")
        elif method == "lineaar":
            return self._linear_opt(bound)
