from coilCalculator import coilCalculator
import matplotlib.pyplot as plt
import pltHelper
import numpy
from scipy.fftpack import fft
from scipy.signal import blackman
import convexApprox
import splinify
import datastore
import solver


def discrete_fprime(f, z):
    pas = z[1] - z[0]
    f1 = numpy.roll(f, -1)
    f0 = numpy.roll(f, 1)
    return (f1 - f0) / (2 * pas)


cas_1 = {
    "Lp": 2,
    "Rp": 5,
    "Lb": 50,
    "Rbi": 6,
    "Rbo": 7,
}

cas_2 = {
    "Lp": 50,
    "Rp": 5,
    "Lb": 2,
    "Rbi": 6,
    "Rbo": 7,
}

cas_1_b = {
    "Lp": 2,
    "Rp": 5,
    "Lb": 50,
    "Rbi": 6,
    "Rbo": 20,
}

cas_2_b = {
    "Lp": 50,
    "Rp": 5,
    "Lb": 2,
    "Rbi": 6,
    "Rbo": 20,
}

cas_3 = {
    "Lp": 10,
    "Rp": 5,
    "Lb": 30,
    "Rbi": 6,
    "Rbo": 9,
}

cas_4 = {
    "Lp": 10,
    "Rp": 5,
    "Lb": 10,
    "Rbi": 6,
    "Rbo": 9,
}


example_1 = (numpy.array([-0.1, -0.0990099, -0.0980198, -0.0970297, -0.0960396,
                          -0.0950495, -0.09405941, -0.09306931, -0.09207921, -0.09108911,
                          -0.09009901, -0.08910891, -0.08811881, -0.08712871, -0.08613861,
                          -0.08514851, -0.08415842, -0.08316832, -0.08217822, -0.08118812,
                          -0.08019802, -0.07920792, -0.07821782, -0.07722772, -0.07623762,
                          -0.07524752, -0.07425743, -0.07326733, -0.07227723, -0.07128713,
                          -0.07029703, -0.06930693, -0.06831683, -0.06732673, -0.06633663,
                          -0.06534653, -0.06435644, -0.06336634, -0.06237624, -0.06138614,
                          -0.06039604, -0.05940594, -0.05841584, -0.05742574, -0.05643564,
                          -0.05544554, -0.05445545, -0.05346535, -0.05247525, -0.05148515,
                          -0.05049505, -0.04950495, -0.04851485, -0.04752475, -0.04653465,
                          -0.04554455, -0.04455446, -0.04356436, -0.04257426, -0.04158416,
                          -0.04059406, -0.03960396, -0.03861386, -0.03762376, -0.03663366,
                          -0.03564356, -0.03465347, -0.03366337, -0.03267327, -0.03168317,
                          -0.03069307, -0.02970297, -0.02871287, -0.02772277, -0.02673267,
                          -0.02574257, -0.02475248, -0.02376238, -0.02277228, -0.02178218,
                          -0.02079208, -0.01980198, -0.01881188, -0.01782178, -0.01683168,
                          -0.01584158, -0.01485149, -0.01386139, -0.01287129, -0.01188119,
                          -0.01089109, -0.00990099, -0.00891089, -0.00792079, -0.00693069,
                          -0.00594059, -0.0049505, -0.0039604, -0.0029703, -0.0019802,
                          -0.0009901, 0., 0.0009901, 0.0019802, 0.0029703,
                          0.0039604, 0.0049505, 0.00594059, 0.00693069, 0.00792079,
                          0.00891089, 0.00990099, 0.01089109, 0.01188119, 0.01287129,
                          0.01386139, 0.01485149, 0.01584158, 0.01683168, 0.01782178,
                          0.01881188, 0.01980198, 0.02079208, 0.02178218, 0.02277228,
                          0.02376238, 0.02475248, 0.02574257, 0.02673267, 0.02772277,
                          0.02871287, 0.02970297, 0.03069307, 0.03168317, 0.03267327,
                          0.03366337, 0.03465347, 0.03564356, 0.03663366, 0.03762376,
                          0.03861386, 0.03960396, 0.04059406, 0.04158416, 0.04257426,
                          0.04356436, 0.04455446, 0.04554455, 0.04653465, 0.04752475,
                          0.04851485, 0.04950495, 0.05049505, 0.05148515, 0.05247525,
                          0.05346535, 0.05445545, 0.05544554, 0.05643564, 0.05742574,
                          0.05841584, 0.05940594, 0.06039604, 0.06138614, 0.06237624,
                          0.06336634, 0.06435644, 0.06534653, 0.06633663, 0.06732673,
                          0.06831683, 0.06930693, 0.07029703, 0.07128713, 0.07227723,
                          0.07326733, 0.07425743, 0.07524752, 0.07623762, 0.07722772,
                          0.07821782, 0.07920792, 0.08019802, 0.08118812, 0.08217822,
                          0.08316832, 0.08415842, 0.08514851, 0.08613861, 0.08712871,
                          0.08811881, 0.08910891, 0.09009901, 0.09108911, 0.09207921,
                          0.09306931, 0.09405941, 0.0950495, 0.0960396, 0.0970297,
                          0.0980198, 0.0990099, 0.1]), numpy.array([3.15297462e-11, 4.32431680e-11, 3.91235410e-11, 4.56562695e-11,
                                                                    5.46413403e-11, 6.01309946e-11, 5.71896744e-11, 6.12869413e-11,
                                                                    7.14918750e-11, 7.84977176e-11, 1.12331396e-10, 1.19929898e-10,
                                                                    1.17054087e-10, 1.23821411e-10, 1.32464215e-10, 1.34261517e-10,
                                                                    1.46376307e-10, 2.13948035e-10, 2.26463172e-10, 2.26561461e-10,
                                                                    2.66821088e-10, 2.62495024e-10, 2.88210869e-10, 3.79189329e-10,
                                                                    3.26483973e-10, 4.54348104e-10, 4.62573499e-10, 5.64379085e-10,
                                                                    7.07050594e-10, 8.39830456e-10, 8.11693752e-10, 9.14334098e-10,
                                                                    1.31007161e-09, 1.24386777e-09, 1.19457236e-09, 1.51449921e-09,
                                                                    2.04709287e-09, 2.03650179e-09, 2.53296770e-09, 2.69178977e-09,
                                                                    4.58180919e-09, 3.87994486e-09, 5.45077014e-09, 6.08526604e-09,
                                                                    9.10566631e-09, 8.01518302e-09, 1.07559988e-08, 1.48623299e-08,
                                                                    2.07009896e-08, 1.83712678e-08, 2.74295176e-08, 3.70084850e-08,
                                                                    3.71594712e-08, 3.27727506e-08, 3.65118511e-08, 6.42185309e-08,
                                                                    1.70479011e-07, 1.15876748e-07, 1.21633813e-07, 1.43680517e-07,
                                                                    1.39649849e-07, 1.92031392e-07, 2.36107360e-07, 3.86835835e-07,
                                                                    4.33539487e-07, 5.35880980e-07, 9.14239294e-07, 1.06931492e-06,
                                                                    1.55061003e-06, 2.26382288e-06, 3.42127741e-06, 5.18619667e-06,
                                                                    7.66420783e-06, 1.22627164e-05, 1.89097803e-05, 2.81976734e-05,
                                                                    3.42485929e-05, 3.16417272e-05, 2.59546637e-05, 2.14243552e-05,
                                                                    1.70104655e-05, 1.43818425e-05, 1.17533962e-05, 9.67061474e-06,
                                                                    7.99037100e-06, 6.61766840e-06, 5.71834441e-06, 5.27480381e-06,
                                                                    3.76124529e-06, 3.41272316e-06, 2.82035612e-06, 2.40723600e-06,
                                                                    2.10181543e-06, 1.68864855e-06, 1.65079749e-06, 8.64456101e-07,
                                                                    7.06980690e-07, 1.30320127e-06, 4.75685828e-07, 1.06861701e-07,
                                                                    -3.92277475e-08, 0.00000000e+00, 3.92277475e-08, -1.06861701e-07,
                                                                    -4.75685828e-07, -1.30320127e-06, -7.06980690e-07, -8.64456101e-07,
                                                                    -1.65079749e-06, -1.68864855e-06, -2.10181543e-06, -2.40723600e-06,
                                                                    -2.82035612e-06, -3.41272316e-06, -3.76124529e-06, -5.27480381e-06,
                                                                    -5.71834441e-06, -6.61766840e-06, -7.99037100e-06, -9.67061474e-06,
                                                                    -1.17533962e-05, -1.43818425e-05, -1.70104655e-05, -2.14243552e-05,
                                                                    -2.59546637e-05, -3.16417272e-05, -3.42485929e-05, -2.81976734e-05,
                                                                    -1.89097803e-05, -1.22627164e-05, -7.66420783e-06, -5.18619667e-06,
                                                                    -3.42127741e-06, -2.26382288e-06, -1.55061003e-06, -1.06931492e-06,
                                                                    -9.14239294e-07, -5.35880980e-07, -4.33539487e-07, -3.86835835e-07,
                                                                    -2.36107360e-07, -1.92031392e-07, -1.39649849e-07, -1.43680517e-07,
                                                                    -1.21633813e-07, -1.15876748e-07, -1.70479011e-07, -6.42185309e-08,
                                                                    -3.65118511e-08, -3.27727506e-08, -3.71594712e-08, -3.70084850e-08,
                                                                    -2.74295176e-08, -1.83712678e-08, -2.07009896e-08, -1.48623299e-08,
                                                                    -1.07559988e-08, -8.01518302e-09, -9.10566631e-09, -6.08526604e-09,
                                                                    -5.45077014e-09, -3.87994486e-09, -4.58180919e-09, -2.69178977e-09,
                                                                    -2.53296770e-09, -2.03650179e-09, -2.04709287e-09, -1.51449921e-09,
                                                                    -1.19457236e-09, -1.24386777e-09, -1.31007161e-09, -9.14334098e-10,
                                                                    -8.11693752e-10, -8.39830456e-10, -7.07050594e-10, -5.64379085e-10,
                                                                    -4.62573499e-10, -4.54348104e-10, -3.26483973e-10, -3.79189329e-10,
                                                                    -2.88210869e-10, -2.62495024e-10, -2.66821088e-10, -2.26561461e-10,
                                                                    -2.26463172e-10, -2.13948035e-10, -1.46376307e-10, -1.34261517e-10,
                                                                    -1.32464215e-10, -1.23821411e-10, -1.17054087e-10, -1.19929898e-10,
                                                                    -1.12331396e-10, -7.84977176e-11, -7.14918750e-11, -6.12869413e-11,
                                                                    -5.71896744e-11, -6.01309946e-11, -5.46413403e-11, -4.56562695e-11,
                                                                    -3.91235410e-11, -4.32431680e-11, -3.15297462e-11]))


def Mu_impact(cas):
    # Mu impact
    Lp = cas["Lp"]
    Rp = cas["Rp"]
    Lb = cas["Lb"]
    Rbi = cas["Rbi"]
    Rbo = cas["Rbo"]
    mu = [5, 10, 50, 100, 500, 1000, 5000]
    n = len(mu)
    res = []
    for k in range(n):
        test = coilCalculator(True, 3)
        test.defineCoil(Lb, Rbi, Rbo)
        test.drawCoil()
        test.defineProjectile(Lp, Rp, mu=mu[k])
        test.drawProjectile()
        test.setSpace()
        test.computeL0()
        test.computedLz(ite=50)
        res += [max(test.dLz)]
        plt.plot(test.dLz_z, test.dLz / max(test.dLz), color=(k / n, 0, 1 - k / n), label=k)
    plt.show()
    # print(mu, res)


def I_impact(cas, mu=200):
    # I impact
    Lp = cas["Lp"]
    Rp = cas["Rp"]
    Lb = cas["Lb"]
    Rbi = cas["Rbi"]
    Rbo = cas["Rbo"]
    i = [5, 10, 50, 100, 500, 1000]
    n = len(i)
    res = []
    for k in range(n):
        test = coilCalculator(True, 3, _i0=i[k])
        test.defineCoil(Lb, Rbi, Rbo)
        test.drawCoil()
        test.defineProjectile(Lp, Rp, mu=mu)
        test.drawProjectile()
        test.setSpace()
        test.computeL0()
        test.computedLz(ite=10)  # dLz is computed via the weigthed force tensor which should be equal to F=1/2*i**2*dLz/dz
        res.append((test.L0, numpy.max(test.dLz)))
        plt.plot(test.dLz_z, test.dLz, color=(k / n, 0, 1 - k / n), label=k)
    print(res)
    plt.show()


def test_case(cas):
    # Mu impact
    Lp = cas["Lp"]
    Rp = cas["Rp"]
    Lb = cas["Lb"]
    Rbi = cas["Rbi"]
    Rbo = cas["Rbo"]
    mu = 100
    test = coilCalculator(True, 3)
    test.defineCoil(Lb, Rbi, Rbo)
    test.drawCoil()
    test.defineProjectile(Lp, Rp, mu=mu)
    test.drawProjectile()
    test.setSpace()
    test.computeL0()
    test.computedLz()
    res = (test.dLz_z, test.dLz)
    # plt.plot(res[0], res[1] / max(test.dLz))
    # plt.show()
    return res


"""
N = len(test.dLz_z)
w = blackman(N)
ywf = fft(test.dLz * w)
xf = numpy.linspace(0, 1 / (2 * (test.dLz_z[1] - test.dLz_z[0])), N / 2)

plt.semilogy(xf[1:N // 2], 2.0 / N * numpy.abs(ywf[1:N // 2]), '-r')
plt.grid()

"""

"""
plt.subplot(1, 2, 1)

c = lFilter(test.dLz_z, test.dLz, test.dLz_nyquist)

# test.computedLz(500, rType="linear")
# a = lFilter(test.dLz_z, test.dLz, test.dLz_nyquist, test.estFreq())
z2 = numpy.linspace(test.dLz_z[0], test.dLz_z[-1], 5000)
plt.plot(test.dLz_z, test.dLz, label="raw 300")
plt.plot(z2, c.dLz()(z2), label="xp auto")
# plt.plot((a.z - a.delay)[a.n_pad:], a.raw_dLz()[a.n_pad:])
# plt.xlabel(r'$z \; (m)$')
# plt.ylabel(r'$\dfrac{dL}{dz} \; (H \cdot m^{-1})$', rotation=0)
pltHelper.centerPlt()

plt.subplot(1, 2, 2)
plt.plot(test.dLz_z, discrete_fprime(test.dLz, test.dLz_z))
plt.plot(z2, c.d2Lz()(z2))
pltHelper.centerPlt()

plt.show()
"""

# Mu_impact(cas_1)
# Mu_impact(cas_1_b)
# Mu_impact(cas_2)
# Mu_impact(cas_2_b)
# Mu_impact(cas_3)
# Mu_impact(cas_4)

# I_impact(cas_1)
# I_impact(cas_1_b)
# I_impact(cas_4)
# I_impact(cas_2)
# I_impact(cas_2_b)
# I_impact(cas_3)


# print(test_case(cas_1))


"""
test_data_convexe_noise = numpy.array([0, 17, 15, 17, 20, 21])
test = convexApprox.Convex_approx_1(test_data_convexe_noise, concave=True, details=True)
plt.plot(test_data_convexe_noise, color=(0, 0, 1))
plt.plot(test.minimize()[0], color=(1, 0, 0))
plt.plot(test.x0, color=(0, 1, 0))
plt.show()

"""

"""
test = convexApprox.Convex_approx(example_1[0], example_1[1], True)
# print(res)
# plt.plot(test._d3Lz[:len(test._d3Lz) // 2])
# plt.plot(test.moving_average(test._d3Lz[:len(test._d3Lz) // 2], 4))
# plt.show()
# print(len(test._d3Lz))

res = test.run_approx()
plt.plot(test.dLz, color=(1, 0, 0))
plt.plot(res)

plt.show()
"""

"""
test_data_bug = numpy.array([3.16417272e-05, 2.59546637e-05, 2.14243552e-05, 1.70104655e-05,
                             1.43818425e-05, 1.17533962e-05, 9.67061474e-06, 7.99037100e-06,
                             6.61766840e-06, 5.71834441e-06, 5.27480381e-06, 3.76124529e-06,
                             3.41272316e-06, 2.82035612e-06, 2.40723600e-06, 2.10181543e-06,
                             1.68864855e-06, 1.65079749e-06, 8.64456101e-07, 7.06980690e-07,
                             1.30320127e-06, 4.75685828e-07, 1.06861701e-07, -3.92277475e-08,
                             0.00000000e+00])
test = convexApprox.Convex_approx_1(test_data_bug, details=True)
plt.plot(test_data_bug, color=(0, 0, 1))
plt.plot(test.minimize()[0], color=(1, 0, 0))
plt.plot(test.x0, color=(0, 1, 0))
plt.show()
"""


def l_construct(cas, space=3):
    # Mu impact
    Lp = cas["Lp"]
    Rp = cas["Rp"]
    Lb = cas["Lb"]
    Rbi = cas["Rbi"]
    Rbo = cas["Rbo"]
    mu = cas['mu']
    test = coilCalculator(True)
    test._coilCalculator__space_factor = space
    test.defineCoil(Lb, Rbi, Rbo)
    test.drawCoil()
    test.defineProjectile(Lp, Rp, mu=mu)
    test.drawProjectile()
    test.setSpace()
    test.computeL0()
    test.computedLz()
    return test


def plot_l(test):
    convex = convexApprox.Convex_approx(test.dLz_z, test.dLz)
    lz = splinify.splinify(test.dLz_z, test.L0, dL=convex.run_approx())

    z = numpy.linspace(lz.z[0], lz.z[-1], 5000)

    ax1 = plt.subplot(311)
    plt.plot(z, lz.Lz()(z), color=(0, 0, 1))
    plt.setp(ax1.get_xticklabels())

    ax2 = plt.subplot(312, sharex=ax1)
    plt.plot(lz.z, test.dLz, color=(1, 0, 0))
    plt.plot(z, lz.dLz()(z), color=(0, 0, 1))
    plt.setp(ax2.get_xticklabels(), visible=False)

    plt.subplot(313, sharex=ax1)
    plt.plot(lz.z, discrete_fprime(test.dLz, test.dLz_z), color=(1, 0, 0))
    plt.plot(z, lz.d2Lz()(z), color=(0, 0, 1))
    plt.show()


def plot_l_b(test):
    convex = convexApprox.Convex_approx(test.dLz_z, test.dLz, order=2)
    print(test.dLz_z, test.dLz)
    # plt.plot(convex._d2Lz)
    # plt.plot(convex.run_approx())
    # plt.show()

    lz = splinify.splinify(test.dLz_z, test.L0, d2L=convex.run_approx())

    z = numpy.linspace(2 * lz.z[0], 2 * lz.z[-1], 10000)

    ax1 = plt.subplot(311)
    plt.plot(z, lz.Lz()(z), color=(0, 0, 1))
    plt.setp(ax1.get_xticklabels())

    ax2 = plt.subplot(312, sharex=ax1)
    plt.plot(lz.z, test.dLz, color=(1, 0, 0))
    plt.plot(z, lz.dLz()(z), color=(0, 0, 1))
    plt.setp(ax2.get_xticklabels(), visible=False)

    plt.subplot(313, sharex=ax1)
    plt.plot(lz.z, discrete_fprime(test.dLz, test.dLz_z), color=(1, 0, 0))
    plt.plot(lz.z, convex.run_approx(), color=(0, 1, 0))
    plt.plot(z, lz.d2Lz()(z), color=(0, 0, 1))
    plt.show()
# l_construct(cas_3)


def solver_test(loc, nb=1):
    coil = datastore.coils.iloc[loc]
    convex = convexApprox.Convex_approx(coil.dLz_z, coil.dLz, order=2)
    lz = splinify.splinify(convex.dLz_z, coil.L0, d2L=convex.run_approx())

    test = solver.gaussSolver(lz, C=0.0047, R=0.1, E=450, m=0.0078)
    res = []
    for i in range(nb):
        res.append(test.solve(-(2.4 * coil.Lb - i) / 2000, 0))
        print("ec" + str(i), test.computeMaxEc(res[i]))
        print("etotal" + str(i), test.computeMaxE(res[i]))
        print("tau" + str(i), test.computeTau(res[i]) * 100)
    test.plot_multiple(res)


def dicho_test(loc):
    coil = datastore.coils.iloc[loc]
    convex = convexApprox.Convex_approx(coil.dLz_z, coil.dLz, order=2)
    lz = splinify.splinify(convex.dLz_z, coil.L0, d2L=convex.run_approx())

    test = solver.gaussSolver(lz, C=0.0047, R=0.1, E=170, m=0.0078)
    res = test.computeOptimal(-(5 * coil.Lb) / 2000)
    print("ec", test.computeMaxEc(res[1]))
    print("etotal", test.computeMaxE(res[1]))
    print("tau", test.computeTau(res[1]) * 100)
    test.plot_single(res[1])
    print(res[0])


def linear_test(loc, plot=False, plot3d=False):
    coil = datastore.coils.iloc[loc]
    convex = convexApprox.Convex_approx(coil.dLz_z, coil.dLz, order=2)
    lz = splinify.splinify(convex.dLz_z, coil.L0, d2L=convex.run_approx())
    if plot:
        plot_l_b(coil)
    test = solver.gaussSolver(lz, C=0.0047, R=0.1 + coil.resistance, E=450, m=0.031)
    res = test._linear_opt(-(5 * coil.Lb) / 2000, plot=plot, plot3d=plot3d)
    # print(res)
    if plot:
        test.plot_single(res[1])
    print(test.computeMaxEc(res[1]), str(int(test.computeTau(res[1]) * 100)) + "%")
    return (test.computeMaxEc(res[1]), str(int(test.computeTau(res[1]) * 100)) + "%")


# solver_test(0, 5)

"""
r = []
for i in range(30):
    r.append(linear_test(1 + i * 31))
for i in r:
    print(i)
# plot_l(datastore.coils.iloc[5])
"""

# linear_test(5, True, True)


def vs_old_maple():
    coil = l_construct({
        'Lp': 22,
        'Rp': 4.5,
        'Lb': 31,
        'Rbi': 6,
        'Rbo': 7,
        'mu': 3.5,
    })
    convex = convexApprox.Convex_approx(coil.dLz_z, coil.dLz, order=2)
    lz = splinify.splinify(convex.dLz_z, coil.L0, d2L=convex.run_approx())
    plot_l_b(coil)
    print("rb", coil.resistance)
    test = solver.gaussSolver(lz, C=0.0050, R=0.016 + 0.075, E=170, m=0.0109)
    res = test._linear_opt(-0.08, plot=True, plot3d=True, epsilon=0.001)
    # print(res)
    test.plot_single(res[1])
    print(test.computeMaxEc(res[1]), str(int(test.computeTau(res[1]) * 10000) / 100) + "%")
    return (test.computeMaxEc(res[1]), str(int(test.computeTau(res[1]) * 100)) + "%")


# vs_old_maple()

def advanced_linear_test(loc, plot=False, plot3d=False):
    coil = datastore.coils.iloc[loc]
    coil = l_construct({
        'Lp': coil.Lp,
        'Rp': coil.Rp,
        'Lb': coil.Lb,
        'Rbi': coil.Rbi,
        'Rbo': coil.Rbo,
        'mu': 100
    }, space=5)
    convex = convexApprox.Convex_approx(coil.dLz_z, coil.dLz, order=2)
    lz = splinify.splinify(convex.dLz_z, coil.L0, d2L=convex.run_approx())
    if plot:
        plot_l_b(coil)
    test = solver.gaussSolver(lz, C=0.0047, R=0.1 + coil.resistance, E=450, m=0.031)
    res = test._linear_opt(-(5 * coil.Lb) / 2000, plot=plot, plot3d=plot3d, epsilon=0.00005)
    # print(res)
    if plot:
        test.plot_single(res[1])
    print(test.computeMaxEc(res[1]), str(int(test.computeTau(res[1]) * 100)) + "%")
    return (test.computeMaxEc(res[1]), str(int(test.computeTau(res[1]) * 100)) + "%")


advanced_linear_test(30, True, True)
