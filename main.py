import coilCalculator
import numpy
import matplotlib.pyplot as plt
# import convexApprox
# import splinify
import datastore
import solver
import convexApprox
import splinify
from multiprocessing import Pool
import pandas as pd
from functools import partial
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from tqdm import tqdm
from scipy.interpolate import griddata


POOL_SIZE = 6


def discrete_fprime(f, z):
    """Discrete derivative

    Discrete derivative of f
    measured at points z.

    Suppose a regular sampling.

    Arguments:
        f {np.array} -- points to derivate
        z {np.array} -- points of measurement

    Returns:
        np.array - the discrete derivative
    """
    pas = z[1] - z[0]
    f1 = numpy.roll(f, -1)
    f0 = numpy.roll(f, 1)
    return (f1 - f0) / (2 * pas)


def coil_construct(coil):
    """Compute a coil inductance

    Given a coil, compute all key
    metrics to perform a simulation:
    - Bare inductance
    - Variation of inductance when projectile moves along the revolution axis
    - Resistance (round wire by default)

    Updates the series and doesn't return anything.

    Arguments:
        coil {pd.Series} -- Mechanical setup
    """
    Lp = coil["Lp"]
    Rp = coil["Rp"]
    Lb = coil["Lb"]
    Rbi = coil["Rbi"]
    Rbo = coil["Rbo"]
    mu = coil["mu"]
    test = coilCalculator.coilCalculator(True, _id=coil.name)
    test.defineCoil(Lb, Rbi, Rbo)
    test.drawCoil()
    test.defineProjectile(Lp, Rp, mu=mu)
    test.drawProjectile()
    test.setSpace()
    test.computeL0()
    test.computedLz()
    coil['L0'] = test.L0
    coil['dLz'] = test.dLz
    coil['dLz_z'] = test.dLz_z
    coil['n_points'] = len(test.dLz_z)
    coil['resistance'] = test.resistance


def build_some_coils(n=10):
    """Compute a batch of coils

    Select n coils in the store that are not computed
    and compute their metrics using multiprocessing.

    Keyword Arguments:
        n {number} -- batch size (default: {10})
    """
    coils = []
    for index, coil in datastore.coils[datastore.coils['dLz'].isnull()][:n].iterrows():
        coils.append(coil)
    with Pool(POOL_SIZE) as p:
        coils = []
        for res in tqdm(p.imap(_build_a_coil, coils), total=len(coils)):
            coils.append(res)
    for coil in coils:
        datastore.update_coil(coil)


def _build_a_coil(coil):
    """ helper for multiprocessing """
    print("Coil", coil.name)
    coil_construct(coil)
    return coil


def find_optimal_launch(loc, C, R, E, v0=0, plot=False, plot3d=False):
    """Compute the optimal launch position

    Given a coil number and an electrical setup,
    computes the optimal launch position of the projectile
    and the key statistics linked (kinetic energy and efficiency).

    Use plot to:
    - plot the coil parameters
    - plot the projection of all launch positions tested
    - plot the dynamic of the best solution

    Use plot3d to:
    - plot the 3d representation of all solutions tested

    Arguments:
        loc {number} -- coil id
        C {number} -- capacity in Farad
        R {number} -- circuit resistance without coil in Ohm
        E {number} -- capacitor tension in Volts

    Keyword Arguments:
        v0 {number} -- starting speed for chained coils (default: {0})
        plot {bool} -- plot 2d informations (default: {False})
        plot3d {bool} -- plot 3d solutions (default: {False})

    Returns:
        tuple - starting position, system's dynamic, output kinetic energy, power efficiency
    """
    coil = datastore.coils.iloc[loc]
    m = numpy.pi * coil.Rp**2 * coil.Lp * 7860 * 10 ** (-9)
    convex = convexApprox.Convex_approx(coil.dLz_z, coil.dLz, order=2)
    lz = splinify.splinify(convex.dLz_z, coil.L0, d2L=convex.run_approx())
    if plot:
        plot_l_b(coil, lz, convex)
    test = solver.gaussSolver(lz, C=C, R=R + coil.resistance, E=E, m=m, v0=v0)
    res = test.computeOptimal(-(5 * coil.Lb) / 2000, plot=plot, plot3d=plot)
    if plot:
        test.plot_single(res[1])
    print("Coil " + str(coil.name) + " opt launch", test.computeMaxEc(res[1]), str(int(test.computeTau(res[1]) * 100)) + "%")
    return (res[0], res[1], test.computeMaxEc(res[1]), test.computeTau(res[1]))


def build_solution(coil_id, setup_id, v0=0, chained=numpy.nan, plot=False):
    """Build a solution

    Given a coil id and an electrical setup id,
    finds the optimal launch position and associated key metrics.

    Arguments:
        coil_id {number} -- coil number in store
        setup_id {number} -- electrical setup in store

    Keyword Arguments:
        v0 {number} -- initial speed (default: {0})
        chained {number} -- last solution id, if the coils are chained. v0 will be the output speed of the last coil (default: {numpy.nan})
        plot {bool} -- plot informations (default: {False})

    Returns:
        pd.Series -- Solution
    """
    if not numpy.isnan(chained):
        v0 = datastore.solutions.iloc[chained].v1
    setup = datastore.setups.iloc[setup_id]
    (z0, dyn, ec, tau) = find_optimal_launch(coil_id, setup.C, setup.R, setup.E, v0=v0, plot=plot, plot3d=plot)
    solution = pd.Series([len(datastore.solutions), setup_id, coil_id, z0, v0, dyn[:, 3][-1], ec, tau, chained],
                         index=['id', 'setup', 'coil', 'z0', 'v0', 'v1', 'Ec', 'tau', 'chained'])
    return solution


def build_some_solutions(setup_id, n=10):
    """Compute a batch of solutions

    Given a setup id, selects n unsolved coils
    settings.

    Uses multiprocessing.

    Arguments:
        setup_id {number} -- setup id in store

    Keyword Arguments:
        n {number} -- batch size (default: {10})
    """
    coil_ids = datastore.coils[datastore.coils['dLz'].notnull()].index.values.tolist()
    existing_sol = datastore.solutions[datastore.solutions['setup'] == setup_id]['coil']
    remaining_coils = numpy.setdiff1d(coil_ids, existing_sol)
    coil_ids = []
    for i in range(n):
        coil_ids.append(remaining_coils[i])
    fun = partial(build_solution, setup_id=setup_id)
    res = []
    with Pool(POOL_SIZE) as p:
        res = []
        for sol in tqdm(p.imap(fun, coil_ids), total=len(coil_ids)):
            res.append(sol)
    for sol in res:
        sol.id = len(datastore.solutions)
        datastore.save_solution(sol)


def plot_l_b(coil, spline, convex):
    """ a helper to plot a coil inductance """
    z = numpy.linspace(2 * spline.z[0], 2 * spline.z[-1], 10000)

    ax1 = plt.subplot(311)
    plt.plot(z, spline.Lz()(z), color=(0, 0, 1))
    plt.setp(ax1.get_xticklabels())

    ax2 = plt.subplot(312, sharex=ax1)
    plt.plot(spline.z, coil.dLz, color=(1, 0, 0))
    plt.plot(z, spline.dLz()(z), color=(0, 0, 1))
    plt.setp(ax2.get_xticklabels(), visible=False)

    plt.subplot(313, sharex=ax1)
    plt.plot(spline.z, convex.run_approx(), color=(0, 1, 0))
    plt.plot(spline.z, discrete_fprime(coil.dLz, coil.dLz_z), color=(1, 0, 0))
    plt.plot(z, spline.d2Lz()(z), color=(0, 0, 1))
    plt.show()


def compute_mu_impact(coil):
    """Check if the Mu approximation is valid

    Arguments:
        coil {pd.Series} -- coil

    Returns:
        pd.Series -- updated coil
    """
    print("Mu", coil.name)
    Lp = coil["Lp"]
    Rp = coil["Rp"]
    Lb = coil["Lb"]
    Rbi = coil["Rbi"]
    Rbo = coil["Rbo"]
    mu = coil["mu"]
    test = coilCalculator.coilCalculator(True)
    test.defineCoil(Lb, Rbi, Rbo)
    test.drawCoil()
    test.defineProjectile(Lp, Rp, mu=mu)
    test.drawProjectile()
    test.setSpace()
    output = test.computeMuImpact()
    if output['valid']:
        coil["mu_approx_valid"] = True
        coil["mu_points"] = output['mus']
        coil["mu_Lz_0"] = output['mu_Lz_0']
    return coil


def compute_some_mu(n=10):
    """Check the mu approximation by batch

    Using multiprocessing.

    Keyword Arguments:
        n {number} -- Batch size (default: {10})
    """
    coils = []
    for index, coil in datastore.coils[datastore.coils['mu_approx_valid'].isnull()][:n].iterrows():
        coils.append(coil)
    with Pool(POOL_SIZE) as p:
        coils = []
        for res in tqdm(p.imap(compute_mu_impact, coils), total=len(coils)):
            coils.append(res)
    for coil in coils:
        datastore.update_coil(coil)


def plot_solutions(setup_id, phi):
    """ plot a solution in 3d, phi should be the wire size """
    df = datastore.solutions[datastore.solutions["setup"] == setup_id].merge(datastore.coils[datastore.coils["phi"] == 1.0], how="inner", left_on="coil", right_index=True)
    df = df[["Lb", "Rbo", "tau"]]

    x1 = numpy.linspace(df['Lb'].min(), df['Lb'].max(), len(df['Lb'].unique()))
    y1 = numpy.linspace(df['Rbo'].min(), df['Rbo'].max(), len(df['Rbo'].unique()))
    x2, y2 = numpy.meshgrid(x1, y1)
    z2 = griddata((df['Lb'], df['Rbo']), numpy.array(df['tau']) * 100, (x2, y2))

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(x2, y2, z2, cmap=cm.viridis,
                           rstride=1, cstride=1,
                           vmin=numpy.nanmin(z2), vmax=numpy.nanmax(z2))

    fig.colorbar(surf, shrink=0.5, aspect=5)
    plt.show()


if __name__ == '__main__':
    # compute_some_mu(10)
    # build_some_coils(10)
    # build_some_solutions(2, 400)
    plot_solutions(2, 1.0)
    # datastore.update_coil(_build_a_coil(datastore.coils.iloc[480]))
    # sol = build_solution(480, 0)
    # sol.id = len(datastore.solutions)
    # datastore.save_solution(sol)
