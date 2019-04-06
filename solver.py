from scipy.integrate import solve_ivp
import numpy
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pltHelper
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from mayavi import mlab
from tqdm import tqdm
import config


class gaussSolver:
    __n = config.SOLVER_TOTAL_STEPS

    def __init__(self, l, C, E, m, R, v0=0):
        """Solver of the dynamic coil-gun system

        Solves the transient state of a coil-gun
        and finds the best initial position.

        Arguments:
            l {splinify} -- generalized inductance
            C {number} -- capacity in Farad
            E {number} -- Initial charge in Volts
            m {number} -- mass of the projectile
            R {number} -- total circuit resistance

        Keyword Arguments:
            v0 {number} -- initial projectile speed for multi-stage coil-guns (default: {0})
        """
        self.l_splin = l
        self.C = C
        self.R = R
        self.E = E
        self.m = m
        self.v0 = v0
        lambd = self.R / (2 * self.l_splin.l0)
        omega_2 = 1 / (self.l_splin.l0 * self.C)
        if lambd**2 <= omega_2:
            # print("osc")
            self._t_max = config.SOLVER_TIME_FACTOR * (1 / lambd)
        elif lambd**2 > omega_2:
            # print("amort")
            self._t_max = config.SOLVER_TIME_FACTOR * (1 / (lambd - numpy.sqrt(lambd**2 - omega_2)))
        self.t = numpy.linspace(0, self._t_max, self.__n)

    def i_eq(self, t, y):
        """Compute the system equations

        Compute all equations to solve the
        current and movement evolution of the system.

        Arguments:
            y {tuple} -- system state in the following order: current, current derivative, position, speed
            t {number} -- time

        Returns:
            tuple -- the derivative of the input : current derivatid, current second order derivative, projectile speed, acceleration
        """
        i, i_p, z, z_p = y
        return [
            i_p,
            -1 / (self.l_splin.Lz()(z) * self.C) * ((self.R + 2 * z_p * self.l_splin.dLz()(z)) * self.C * i_p + (1 + self.C * (i**2 / self.m / 2 * self.l_splin.dLz()(z)**2 + z_p**2 * self.l_splin.d2Lz()(z))) * i),
            z_p,
            i**2 / self.m / 2 * self.l_splin.dLz()(z),
        ]

    def y0(self, z0, z_p0):
        """Initial condition

        Compute the initial condition of the system

        Arguments:
            z0 {number} -- initial position
            z_p0 {number} -- initial speed

        Returns:
            list - initial current, current derivative, initial position, initial speed
        """
        return [
            0,
            -self.E / self.l_splin.Lz()(z0),
            z0,
            z_p0,
        ]

    def solve(self, z0, z0_p):
        """Compute system dynamic

        Solve the dynamic ODE

        Arguments:
            z0 {number} -- initial position
            z0_p {number} -- initial speed

        Returns:
            np.array -- (current, current derivative, position, speed) * timesteps(see odeint documentation)
        """
        return solve_ivp(self.i_eq, (0, self._t_max), self.y0(z0, z0_p), t_eval=self.t).y

    def plot_single(self, result):
        """Plot the dynamic

        Given the result of the solve method,
        plots the evolution of current in the coil
        and the evolution of projectile's speed.

        Arguments:
            result {np.array} -- odeint solution
        """
        plt.subplots_adjust(hspace=0.8)
        ax1 = plt.subplot(211)
        plt.plot(self.t, result[0, :], color=(0, 0, 1), label="i(t)")
        ax1.set_title(r"$i(t)$", fontsize=11)
        ax1.set(xlabel=r"$s$", ylabel=r"$A$")
        plt.setp(ax1.get_xticklabels())

        ax2 = plt.subplot(212, sharex=ax1)
        plt.plot(self.t, result[3, :], color=(0, 0, 1), label="dz/dt(t)")
        ax2.set_title(r"$\mathrm{Projectile \ speed \ vs. \ time}$", fontsize=11)
        ax2.set(xlabel=r"$s$", ylabel=r"$m.s^{-1}$")
        plt.show()

    def plot_multiple(self, results):
        """Plot a batch of results

        Plot all results for different starting position.

        Arguments:
            results {np.array} -- list of simulations
        """
        n = len(results)
        plt.subplots_adjust(hspace=0.8)
        ax1 = plt.subplot(211)
        for i in range(n):
            plt.plot(self.t, results[i][0, :], color=(i / n, 0, 1 - i / n), label="i(t) - " + str(i))
        plt.setp(ax1.get_xticklabels())
        ax1.set_title(r"$i(t)$", fontsize=11)
        ax1.set(xlabel=r"$s$", ylabel=r"$A$")

        ax2 = plt.subplot(212, sharex=ax1)
        for i in range(n):
            plt.plot(self.t, results[i][3, :], color=(i / n, 0, 1 - i / n), label="dz/dt(t) - " + str(i))
        ax2.set_title(r"$\mathrm{Projectile \ speed \ vs. \ time}$", fontsize=11)
        ax2.set(xlabel=r"$s$", ylabel=r"$m.s^{-1}$")

        plt.show()

    def computeMaxEc(self, result):
        """Compute the maximum kinetic energy

        Max kinetic energy of the projectile.
        The OUTPUT kinetic energy in reality.

        Arguments:
            result {np.array} -- odeint solution

        Returns:
            number -- projectile's output kinetic energy
        """
        v_max = result[3, -1]
        # print(v_max)
        return 1 / 2 * self.m * v_max**2

    def computeMaxE(self, result):
        """Compute initial energy

        Energy initially stored in the capacitor bank

        Arguments:
            result {np.array} -- odeint solution

        Returns:
            number -- capacitor initial energy
        """
        return 1 / 2 * self.C * self.E**2

    def computeTau(self, result):
        """Energy transfer

        Compute the percentage of power
        that was transferred to the projectile
        in one shot.

        Arguments:
            result {np.array} -- odeint solution

        Returns:
            number -- efficiency of the shot
        """
        return self.computeMaxEc(result) / self.computeMaxE(result)

    def _linear_opt(self, bound, epsilon=config.SOLVER_OPT_STEP, plot=False, plot3d=False):
        """Find optimal launch position with linear search

        Compute bound / epsilon solutions with different initial
        position and returns the best initial position and dynamic.

        Arguments:
            bound {number} -- largest position to try

        Keyword Arguments:
            epsilon {number} -- linear step (default: {config.SOLVER_OPT_STEP})
            plot {bool} -- plot solutions (default: {False})
            plot3d {bool} -- plot3d surface (default: {False})

        Returns:
            tuple -- best initial position and associated dynamic (current, current variation, position, speed)
        """
        res = []
        z0 = []
        i = 0

        n = int(numpy.abs(bound / epsilon))
        for i in tqdm(range(n), disable=not plot):
            z0.append(bound + epsilon * i)
            res.append(self.solve(bound + epsilon * i, self.v0))
        res = numpy.array(res)

        if plot:
            colors = []
            plt.subplots_adjust(hspace=0.8)
            ax1 = plt.subplot(211)
            for i in range(n):
                colors.append((i / n, 0, 1 - i / n))
                plt.plot(self.t, res[i, 3, :], color=(i / n, 0, 1 - i / n), label="dz/dt(t) - " + str(i))
            plt.setp(ax1.get_xticklabels())
            ax1.set_title(r"$\mathrm{Projectile \ speed \ vs. \ time}$", fontsize=11)
            ax1.set(xlabel=r"$m.s^{-1}$", ylabel=r"$s$")

            ax2 = plt.subplot(212)
            line = []
            for k in range(len(z0)):
                line.append([z0[k], res[k, 3, -1]])
            line = numpy.array(line).reshape(-1, 1, 2)
            line = numpy.hstack([line[:-1], line[1:]])
            coll = LineCollection(line, colors=colors)
            ax2.add_collection(coll)
            ax2.autoscale_view()
            ax2.set_title(r"$\mathrm{Projectile \ speed \ vs. \ initial \ position}$", fontsize=11)
            ax2.set(xlabel=r"$m.s^{-1}$", ylabel=r"$m$")
            plt.show()

        if plot3d:
            """
            fig = plt.figure()
            ax = fig.gca(projection='3d')
            X, Y = numpy.meshgrid(self.t, z0)
            # print(numpy.shape(X))
            # print(numpy.shape(Y))
            # print(numpy.shape(res))
            ax.plot_surface(X, Y, res[:, :, 3], cmap=cm.viridis, linewidth=0, antialiased=False, rcount=200, ccount=200)
            plt.show()
            """

            X, Y = numpy.meshgrid(self.t, z0)
            im = res[:, 3, :]

            """
            # fig = plt.figure()
            # x = fig.add_subplot(111, projection='3d')
            mlab.figure(bgcolor=(1, 1, 1), fgcolor=(0, 0, 0))
            src = mlab.pipeline.array2d_source(im)
            warp = mlab.pipeline.warp_scalar(src)
            normals = mlab.pipeline.poly_data_normals(warp)
            surf = mlab.pipeline.surface(normals, colormap="viridis")
            x_scale = numpy.max(self.t)
            y_scale = numpy.abs(numpy.max(z0) - numpy.min(z0))
            z_scale = numpy.abs(numpy.max(numpy.array(res[:, :, 3])[:, -1]) - numpy.min(numpy.array(res[:, :, 3])[:, -1]))
            max_scale = numpy.max([x_scale, y_scale])  # , z_scale])
            # print([x_scale, y_scale, z_scale])
            # print(max_scale)
            # print((1.0 * x_scale / max_scale, 1.0 * y_scale / max_scale, 1.0))
            surf.actor.actor.scale = (1, 0.1, 1)
            axes = mlab.axes(surf)
            axes.label_text_property.font_size = 10
            axes.label_text_property.font_family = 'courier'
            mlab.show()
            """
            fig = plt.figure()
            ax1 = fig.add_subplot(1, 1, 1)
            plt.contourf(X, Y, im, 100, cmap=cm.viridis)
            ax1.set_xlabel(r"$\mathrm{Time \ }(s)$")
            ax1.set_ylabel(r"$\mathrm{Initial \ position \ }(m)$")
            cbar = plt.colorbar()
            cbar.ax.set_ylabel(r"$\mathrm{Projectile \ speed \ }(m.s^{-1})$")
            plt.show()

        # print(res[:, :, 3])
        arg = numpy.argmax(res[:, 3, -1])
        return (z0[arg], res[arg])

    def computeOptimal(self, bound, method="linear", plot=False, plot3d=False):
        """Find optimal launch position

        Different search method could be implemented

        Arguments:
            bound {number} -- largest position to try

        Keyword Arguments:
            method {str} -- search method (default: {"linear"})
            plot {bool} -- plot all dynamics (default: {False})
            plot3d {bool} -- plot 3d surface (default: {False})

        Returns:
            tuple -- best initial position and associated dynamic (current, current variation, position, speed)

        Raises:
            ValueError -- unkown search method
        """
        if method not in ["dicho", "linear"]:
            raise ValueError("Only linear opt available.")
        elif method == "linear":
            return self._linear_opt(bound, plot=plot, plot3d=plot)
