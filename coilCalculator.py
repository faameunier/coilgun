import femm
import numpy
from progbar import progbar


class coilCalculator:
    __nyq_secu = 1.01

    def __init__(self, bHide=False, meshsize=1, _i0=100):
        self.meshsize = meshsize
        self.Lb = None
        self.Rbi = None
        self.Rbo = None
        self.phi = None
        self.rho = None
        self.n = None
        self.resistance = None
        self.Lp = None
        self.Rp = None
        self.m_vol_fer = None
        self.mu = None
        self.mass = None
        self.espace = None
        self.wire_type = None
        self._i0 = _i0
        femm.openfemm(bHide)
        femm.create(0)
        femm.mi_probdef(0, "millimeters", "axi", 1E-16)
        femm.mi_saveas("temp/temp.fem")
        femm.mi_addcircprop("Bobine", self._i0, 1)
        femm.mi_addmaterial("Air", 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0)

    def defineCoil(self, Lb, Rbi, Rbo, phi=1, rho=17 * 10**(-9), wire_type="round"):
        if wire_type not in ["round", "square"]:
            raise BaseException("Wire should be round or square.")
        else:
            self.wire_type = wire_type
        if self.espace is None:
            if Rbo >= Rbi + phi and Lb >= phi and Rbi > 0 and phi > 0 and rho >= 0:
                if self.Rp is None or Rbi >= self.Rp:
                    self.Lb = Lb
                    self.Rbi = Rbi
                    self.Rbo = Rbo
                    self.phi = phi
                    self.rho = rho
                    self.n = Lb / phi * (Rbo - Rbi) / phi
                    self.resistance = 4 * rho * (Rbo**2 - Rbi**2) * Lb / phi**4
                    self.deleteCoil()
                else:
                    raise BaseException("Impossible coil/projectile geometry.")
            else:
                raise BaseException("Impossible coil geometry.")
        else:
            raise BaseException("Space already defined.")

    def deleteCoil(self):
        femm.mi_clearselected()
        femm.mi_selectgroup(2)
        femm.mi_deleteselected()
        femm.mi_deletematerial("Cuivre")

    def drawCoil(self):
        if self.Lb is not None:
            femm.mi_clearselected()
            femm.mi_addmaterial("Cuivre", 1, 1, 0, 0, 1 / self.rho * 10**11, 0, 0, 1, 3 if self.wire_type == "round" else 6, 0, 0, 1, self.phi)
            femm.mi_addnode(self.Rbi, -self.Lb / 2)
            femm.mi_addnode(self.Rbo, -self.Lb / 2)
            femm.mi_addnode(self.Rbi, self.Lb / 2)
            femm.mi_addnode(self.Rbo, self.Lb / 2)
            femm.mi_addsegment(self.Rbi, -self.Lb / 2, self.Rbo, -self.Lb / 2)
            femm.mi_addsegment(self.Rbo, -self.Lb / 2, self.Rbo, self.Lb / 2)
            femm.mi_addsegment(self.Rbo, self.Lb / 2, self.Rbi, self.Lb / 2)
            femm.mi_addsegment(self.Rbi, -self.Lb / 2, self.Rbi, self.Lb / 2)
            femm.mi_selectnode(self.Rbi, -self.Lb / 2)
            femm.mi_selectnode(self.Rbo, -self.Lb / 2)
            femm.mi_selectnode(self.Rbi, self.Lb / 2)
            femm.mi_selectnode(self.Rbo, self.Lb / 2)
            femm.mi_selectsegment(self.Rbi, 0)
            femm.mi_selectsegment((self.Rbi + self.Rbo) / 2, -self.Lb / 2)
            femm.mi_selectsegment(self.Rbo, 0)
            femm.mi_selectsegment((self.Rbi + self.Rbo) / 2, self.Lb / 2)
            femm.mi_setgroup(2)
            femm.mi_addblocklabel((self.Rbi + self.Rbo) / 2, 0)
            femm.mi_selectlabel((self.Rbi + self.Rbo) / 2, -self.Lb / 2)
            femm.mi_setblockprop("Cuivre", 0, self.meshsize, "Bobine", 0, 2, self.n)
            femm.mi_clearselected()
        else:
            raise BaseException("No coil defined.")

    def defineProjectile(self, Lp, Rp, m_vol_fer=7800, mu=100):
        if self.espace is None:
            if Lp > 0 and Rp > 0 and m_vol_fer >= 0 and mu >= 1:
                if self.Rbi is None or self.Rbi >= Rp:
                    self.Lp = Lp
                    self.Rp = Rp
                    self.m_vol_fer = m_vol_fer
                    self.mu = mu
                    self.mass = numpy.pi * Rp ** 2 * Lp * m_vol_fer * 10 ** (-9)
                    self.deleteProjectile()
                else:
                    raise BaseException("Impossible coil/projectile geometry.")
            else:
                raise BaseException("Impossible coil geometry.")
        else:
            raise BaseException("Space already defined.")

    def deleteProjectile(self):
        femm.mi_clearselected()
        femm.mi_selectgroup(1)
        femm.mi_deleteselected()
        femm.mi_deletematerial("Projectile")
        if self.espace is not None:
            femm.mi_addsegment(0, -self.espace, 0, self.espace)

    def drawProjectile(self):
        if self.Lp is not None:
            femm.mi_addmaterial("Projectile", self.mu, self.mu, 0, 0, 0, 0, 0, 1, 0, 0, 0)
            femm.mi_clearselected()
            femm.mi_addnode(0, -self.Lp / 2)
            femm.mi_addnode(self.Rp, -self.Lp / 2)
            femm.mi_addnode(0, self.Lp / 2)
            femm.mi_addnode(self.Rp, self.Lp / 2)
            femm.mi_addsegment(0, -self.Lp / 2, self.Rp, -self.Lp / 2)
            femm.mi_addsegment(self.Rp, -self.Lp / 2, self.Rp, self.Lp / 2)
            femm.mi_addsegment(self.Rp, self.Lp / 2, 0, self.Lp / 2)
            femm.mi_addsegment(0, self.Lp / 2, 0, -self.Lp / 2)
            femm.mi_selectnode(0, -self.Lp / 2)
            femm.mi_selectnode(self.Rp, -self.Lp / 2)
            femm.mi_selectnode(0, self.Lp / 2)
            femm.mi_selectnode(self.Rp, self.Lp / 2)
            femm.mi_selectsegment(0, 0)
            femm.mi_selectsegment(self.Rp / 2, -self.Lp / 2)
            femm.mi_selectsegment(self.Rp, 0)
            femm.mi_selectsegment(self.Rp / 2, self.Lp / 2)
            femm.mi_setgroup(1)
            femm.mi_addblocklabel(self.Rp / 2, 0)
            femm.mi_selectlabel(self.Rp / 2, 0)
            femm.mi_setblockprop("Projectile", 0, self.meshsize, "<None>", 0, 1, 0)
            femm.mi_clearselected()
        else:
            raise BaseException("No projectile defined.")

    def setSpace(self):
        if self.Lp is not None and self.Lb is not None:
            femm.mi_clearselected()
            self.espace = 3 * max(self.Lb, self.Rbo, self.Lp)
            femm.mi_addblocklabel(2 * self.Rbo, 0)
            femm.mi_selectlabel(2 * self.Rbo, 0)
            femm.mi_setblockprop("Air", 0, self.meshsize, "<None>", 0, 3, 0)
            femm.mi_makeABC(7, self.espace, 0, 0, 0)
            femm.mi_zoomnatural()
        else:
            raise BaseException("Define coil and projectile first.")

    def computeL0(self):
        self.deleteProjectile()
        femm.mi_refreshview()
        femm.mi_analyze()
        femm.mi_loadsolution()
        # print(femm.mo_getcircuitproperties("Bobine"))
        self.L0 = femm.mo_getcircuitproperties("Bobine")[2] / self._i0
        femm.mo_close()
        self.drawProjectile()

    def computedLz(self, ite=0, rType="linear"):
        # print(self.__i0)
        self.deleteProjectile()
        self.drawProjectile()
        (pas, pos, ite) = self._compute_range(ite, rType)
        force = numpy.zeros(ite)
        femm.mi_selectgroup(1)
        femm.mi_movetranslate2(0, pos[0], 4)
        for i in range(ite // 2):
            progbar(i, ite // 2 - 1, 10)
            femm.mi_analyze()
            femm.mi_loadsolution()
            femm.mo_groupselectblock(1)
            force[i] = femm.mo_blockintegral(19)
            force[ite - i - 1] = -force[i]
            femm.mi_selectgroup(1)
            femm.mi_movetranslate2(0, pas[i], 4)
        self.dLz = 2 * force / self._i0**2
        # print("dLz", self.dLz)
        self.dLz_z = pos * 10**-3
        self.dLz_nyquist = 1 / (2 * numpy.mean(pas) * 10**-3)

    def _compute_range(self, ite=0, rType="linear"):
        x_max = 2 / 3 * self.espace
        if ite != 0 and ite % 2 == 0:
            ite += 1
        if rType == "linear":
            if ite == 0:
                ite = numpy.int(numpy.ceil(4 * self.estFreq() * self.__nyq_secu * 10**-3 * x_max + 1))
                if ite % 2 == 0:
                    ite += 1
            pos = numpy.array([x_max * (-1 + 2 * i / (ite - 1)) for i in range(ite)])
            pas = numpy.ones(ite) * 2 * x_max / (ite - 1)
            # print("pos", pos)
            # print("pas", pas)
            # print("n", ite)
            # print("mean pas", numpy.mean(pas))
            # print("1/f", 1 / self.estFreq() * 10**3)
            return (pas, pos, ite)
        if rType == "tchebychev":
            if ite == 0:
                ite = numpy.int(numpy.ceil(4 * self.estFreq() * self.__nyq_secu * 10**-3 * x_max + 1))  # average distance for n iterations is  2 / (n - 1) * np.sin((n - 1) / 2 / n * np.pi)
                if ite % 2 == 0:
                    ite += 1
            pos = numpy.zeros(ite)
            pas = numpy.zeros(ite)
            pos[0] = x_max * (numpy.cos((1 / 2 / ite - 1) * numpy.pi))
            for i in range(2, ite + 1):
                pos[i - 1] = x_max * (numpy.cos(((2 * i - 1) / 2 / ite - 1) * numpy.pi))
                pas[i - 2] = pos[i - 1] - pos[i - 2]
            pos[ite // 2] = 0
            pas[ite // 2 - 1] = -pos[ite // 2 - 1]
            pas[ite // 2] = pos[ite // 2 + 1]
            # print("pos", pos)
            # print("pas", pas)
            # print("n", ite)
            # print("mean pas", numpy.mean(pas))
            # print("1/f", 1 / self.estFreq() * 10**3)
            return (pas, pos, ite)
        else:
            raise BaseException("Unknown node computation type.")

    def computeMuImpact(self, mus=[1, 5, 10, 50, 100, 500, 1000, 5000], error=0.05):
        _mu = self.mu
        res = []
        test_res = []
        for mu in mus:
            self.mu = mu
            self.deleteProjectile()
            self.drawProjectile()
            femm.mi_clearselected()
            femm.mi_selectgroup(1)

            femm.mi_analyze()
            femm.mi_loadsolution()
            femm.mo_groupselectblock(1)
            res.append(femm.mo_getcircuitproperties("Bobine")[2] / self._i0)

            femm.mi_movetranslate2(0, self.Lb / 2, 4)
            femm.mi_analyze()
            femm.mi_loadsolution()
            femm.mo_groupselectblock(1)
            test_res.append(femm.mo_getcircuitproperties("Bobine")[2] / self._i0)
        self.mu = _mu
        success = True
        for i in range(1, len(test_res)):
            if numpy.abs((res[i] / res[0]) / (test_res[i] / test_res[0]) - 1) > error:
                success = False
                break
        if success:
            return (mus, res)
        else:
            return False

    def estFreq(self):
        return 2 / (min(self.Lb, self.Lp) * 10**-3)
