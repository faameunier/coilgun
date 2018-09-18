import coilCalculator
# import numpy
# import convexApprox
# import splinify
import datastore


def coil_construct(coil):
    # Mu impact
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
    # convex = convexApprox.Convex_approx(test.dLz_z, test.dLz)
    # lz = splinify.splinify(test.dLz_z, convex.run_approx(), test.L0)
    coil['L0'] = test.L0
    coil['dLz'] = test.dLz
    coil['dLz_z'] = test.dLz_z
    coil['n_points'] = len(test.dLz_z)
    # coil['splinify'] = lz
    coil['resistance'] = test.resistance


def build_some_coils(n=10):
    i = 0
    for index, coil in datastore.coils[datastore.coils['dLz'].isnull()].iterrows():
        print(index)
        coil_construct(coil)
        datastore.update_coil(coil)
        # datastore.save_all()
        i += 1
        if i == n:
            break


build_some_coils(30)
