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


def discrete_fprime(f, z):
    pas = z[1] - z[0]
    f1 = numpy.roll(f, -1)
    f0 = numpy.roll(f, 1)
    return (f1 - f0) / (2 * pas)


def coil_construct(coil):
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
    test.computeL0()
    test.computedLz()
    coil['L0'] = test.L0
    coil['dLz'] = test.dLz
    coil['dLz_z'] = test.dLz_z
    coil['n_points'] = len(test.dLz_z)
    coil['resistance'] = test.resistance


def build_some_coils(n=10):
    coils = []
    for index, coil in datastore.coils[datastore.coils['dLz'].isnull()][:n].iterrows():
        coils.append(coil)
    # print(coils)
    with Pool(8) as p:
        coils = p.map(_build_a_coil, coils)
        # coil_construct(coil)
    for coil in coils:
        datastore.update_coil(coil)
        # datastore.save_all()


def _build_a_coil(coil):
    print(coil.name)
    coil_construct(coil)
    return coil


def find_optimal_launch(loc, C, R, E, v0=0, plot=False, plot3d=False):
    coil = datastore.coils.iloc[loc]
    m = numpy.pi * coil.Rp**2 * coil.Lp * 7860 * 10 ** (-9)
    convex = convexApprox.Convex_approx(coil.dLz_z, coil.dLz, order=2)
    lz = splinify.splinify(convex.dLz_z, coil.L0, d2L=convex.run_approx())
    if plot:
        plot_l_b(coil, lz, convex)
    test = solver.gaussSolver(lz, C=C, R=R + coil.resistance, E=E, m=m, v0=v0)
    res = test.computeOptimal(-(5 * coil.Lb) / 2000, plot=plot, plot3d=plot)
    # print(res)
    if plot:
        test.plot_single(res[1])
    print(test.computeMaxEc(res[1]), str(int(test.computeTau(res[1]) * 100)) + "%")
    return (res[0], res[1], test.computeMaxEc(res[1]), test.computeTau(res[1]))


def build_solution(setup_id, coil_id, v0=0, chained=numpy.nan, plot=False):
    if not numpy.isnan(chained):
        v0 = datastore.solutions.iloc[chained].v1
    setup = datastore.setups.iloc[setup_id]
    (z0, dyn, ec, tau) = find_optimal_launch(coil_id, setup.C, setup.R, setup.E, v0=v0, plot=plot, plot3d=plot)
    solution = pd.Series([len(datastore.solutions), setup_id, coil_id, z0, v0, dyn[:, 3][-1], ec, tau, chained],
                         index=['id', 'setup', 'coil', 'z0', 'v0', 'v1', 'Ec', 'tau', 'chained'])
    datastore.save_solution(solution)


def build_some_solutions(setup_id, n=10):
    coil_ids = datastore.coils.index.values.tolist()
    print(datastore.solutions[datastore.solutions['setup'] == setup_id])
    existing_sol = datastore.solutions[datastore.solutions['setup'] == setup_id]['coil']
    remaining_coils = numpy.setdiff1d(coil_ids, existing_sol)
    coil_ids = []
    for i in range(n):
        coil_ids.append((setup_id, remaining_coils[i]))
    with Pool(8) as p:
        p.map(build_solution, coil_ids)


def plot_l_b(coil, spline, convex):
    print(coil.dLz_z, coil.dLz)
    # plt.plot(convex._d2Lz)
    # plt.plot(convex.run_approx())
    # plt.show()

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


def compute_mu_impact(coil, full_print=False):
    print(coil.name)
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
    coils = []
    for index, coil in datastore.coils[datastore.coils['mu_approx_valid'].isnull()][:n].iterrows():
        coils.append(coil)
    # print(coils)
    with Pool(8) as p:
        coils = p.map(compute_mu_impact, coils)
        # coil_construct(coil)
    for coil in coils:
        datastore.update_coil(coil)
        # datastore.save_all()


if __name__ == '__main__':
    # compute_some_mu(100)
    # build_some_coils(15)
    build_some_solutions(0, 10)
# build_a_coil(800)
# find_optimal_launch(800, C=0.0024, E=400, R=0.07, plot=True, plot3d=False)
# find_optimal_launch(10, C=0.0024, E=400, R=0.07, plot=True, plot3d=True)
