import femm
import numpy
from tqdm import tqdm
import os


class coilCalculator:
    """ The aim of this class to compute, using finite element methods, some key data around a given coilgun problem """
    __nyq_secu = 1.01
    __space_factor = 5

    def __init__(self, bHide=False, meshsize=1, _i0=100, _id=None):
        """Initialize a FEMM solver

        Open FEMM and define the key elements of the magnetic problem.
        The problem is considered STATIC.
        All measurements are in MILLIMETERS here !

        Keyword Arguments:
            bHide {bool} -- Either to show or hide the window. Hiding it provides a nice speed improvement (default: {False})
            meshsize {number} -- Size of the mesh used for the finite elements. The smaller the more precise, but it comes with a computational cost (default: {1})
            _i0 {number} -- Current used in computation. Any value should do, we use 100 to avoid working with very small floats (default: {100})
            _id {number} -- Some id used to save the problem, if one requires to track and come back to the FEMM model (default: {None})
        """
        if _id is not None:
            self._seed = str(_id)
        else:
            self._seed = str(numpy.random.randint(10000))
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
        femm.mi_saveas("temp/temp" + self._seed + ".fem")
        femm.mi_addcircprop("Bobine", self._i0, 1)
        femm.mi_addmaterial("Air", 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0)

    def defineCoil(self, Lb, Rbi, Rbo, phi=1, rho=17 * 10**(-9), wire_type="round"):
        """Define the coil

        Define the coil in FEMM.
        All measurements are in millimeters.

        Arguments:
            Lb {number} -- Length of the coil
            Rbi {number} -- Inside radius of the coil
            Rbo {number} -- Outside radius of the coil

        Keyword Arguments:
            phi {number} -- diameter of the wire (default: {1})
            rho {number} -- resistivity of the wire, defaults to copper (default: {17 * 10**(-9)})
            wire_type {str} -- type of wire, can either be round or square (this has some impact on the model, so be sure to select wisely) (default: {"round"})

        Raises:
            ValueError -- Something went wrong, please check the error for more details
        """
        if wire_type not in ["round", "square"]:
            raise ValueError("Wire should be round or square.")
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
                    self.resistance = 4 * rho * ((Rbo * 10**-3)**2 - (Rbi * 10**-3)**2) * (Lb * 10**-3) / (phi * 10**-3)**4
                    self.deleteCoil()
                else:
                    raise ValueError("Impossible coil/projectile geometry.")
            else:
                raise ValueError("Impossible coil geometry.")
        else:
            raise ValueError("Space already defined.")

    def deleteCoil(self):
        """Delete the coil

        Provided for debugging, should not be used.
        """
        femm.mi_clearselected()
        femm.mi_selectgroup(2)
        femm.mi_deleteselected()
        femm.mi_deletematerial("Cuivre")

    def drawCoil(self):
        """Draw the coil

        Draws the coil in the FEMM instance

        Raises:
            Exception -- The coil is not defined, please call defineCoil before.
        """
        if self.Lb is not None:
            femm.mi_clearselected()
            femm.mi_addmaterial("Cuivre", 1, 1, 0, 0, 1 / self.rho * 10**-6, 0, 0, 1, 3 if self.wire_type == "round" else 6, 0, 0, 1, self.phi)
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
            raise Exception("No coil defined.")

    def defineProjectile(self, Lp, Rp, m_vol_fer=7800, mu=100):
        """Define projectile properties

        Define the projectile properties.
        All measurement in millimeters.

        Arguments:
            Lp {number} -- Length of the projectile
            Rp {number} -- Radius of the projectile

        Keyword Arguments:
            m_vol_fer {number} -- true density of the projectile (default: {7800})
            mu {number} -- magnetic susceptibility of the material. This should be set with excessive care. The model properties are not linear when it comes to mu. There's still work in progress here. (default: {100})

        Raises:
            ValueError -- Something went wrong, check the error.
        """
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
                    raise ValueError("Impossible coil/projectile geometry.")
            else:
                raise ValueError("Impossible coil geometry.")
        else:
            raise ValueError("Space already defined.")

    def deleteProjectile(self):
        """Delete the projectile

        Deletes the projectile (drawn) but doesn't erase its properties.
        """
        femm.mi_clearselected()
        femm.mi_selectgroup(1)
        femm.mi_deleteselected()
        femm.mi_deletematerial("Projectile")
        if self.espace is not None:
            femm.mi_addsegment(0, -self.espace, 0, self.espace)

    def drawProjectile(self):
        """Draw projectile

        Draws the projectil in the FEMM instance

        Raises:
            Exception -- Projectile is not defined
        """
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
            raise Exception("No projectile defined.")

    def setSpace(self):
        """Define space

        Define the whole space used in FEMM

        Raises:
            Exception -- Coil and projectile must be defined first to compute a safe space size.
        """
        if self.Lp is not None and self.Lb is not None:
            femm.mi_clearselected()
            self.espace = self.__space_factor * max(self.Lb, self.Rbo, self.Lp)
            femm.mi_addblocklabel(2 * self.Rbo, 0)
            femm.mi_selectlabel(2 * self.Rbo, 0)
            femm.mi_setblockprop("Air", 0, self.meshsize, "<None>", 0, 3, 0)
            femm.mi_makeABC(7, self.espace, 0, 0, 0)
            femm.mi_zoomnatural()
        else:
            raise Exception("Define coil and projectile first.")

    def computeL0(self):
        """Compute L0

        Compute the bare inductance of the coil without projectile.
        """
        self.deleteProjectile()
        femm.mi_refreshview()
        femm.mi_analyze()
        femm.mi_loadsolution()
        # print(femm.mo_getcircuitproperties("Bobine"))
        self.L0 = femm.mo_getcircuitproperties("Bobine")[2] / self._i0
        self.resistance = femm.mo_getcircuitproperties("Bobine")[1] / self._i0
        femm.mo_close()
        self.drawProjectile()

    def computedLz(self, ite=0, rType="linear"):
        """Compute dLz

        Compute the variation of inductance while the
        projectile moves on the axis. If ite is zero,
        some guess is made about the number of iterations required
        for decent approximation.
        By default the projectile is moved linearly, but it
        is possible to set the movement type to tchebychev in order
        to minimize the Runge phenomenom. However not all the code
        is compatible with it.

        Rather than computing the variation of inductance, we compute
        the force on the projectile and correct it (explanations are available
        somewhere on this git :) ).

        Keyword Arguments:
            ite {number} -- number of steps (default: {0})
            rType {str} -- type of movement, linear or tchebychev (default: {"linear"})
        """
        self.deleteProjectile()
        self.drawProjectile()
        (pas, pos, ite) = self._compute_range(ite, rType)
        force = numpy.zeros(ite)
        femm.mi_selectgroup(1)
        femm.mi_movetranslate2(0, pos[0], 4)
        for i in tqdm(range(ite // 2)):
            femm.mi_analyze()
            femm.mi_loadsolution()
            femm.mo_groupselectblock(1)
            force[i] = femm.mo_blockintegral(19)
            force[ite - i - 1] = -force[i]
            femm.mi_selectgroup(1)
            femm.mi_movetranslate2(0, pas[i], 4)
        self.dLz = 2 * force / self._i0**2
        self.dLz_z = pos * 10**-3
        self.dLz_nyquist = 1 / (2 * numpy.mean(pas) * 10**-3)

    def _compute_range(self, ite=0, rType="linear"):
        """Compute the projectile range of movement

        The projectile starting point is 2/3 of the space size.
        The final position is in the middle of the coil.

        linear computes a linear scale of movements.
        tchebychev computes an optimized path to minimize
        Runge's effect by computing the Tchebychev nodes.

        If ite is not specified, it is guessed based on several informations.
        First we use estFreq which is a guess of "how fast" the inductance can change (empirical).
        Then we chose a number of steps that is compatible with Shannon theorem's and again some
        additionnal margin.

        Keyword Arguments:
            ite {number} -- Number of iteration, if 0, it is guessed (default: {0})
            rType {str} -- linear or tchebychev (default: {"linear"})

        Raises:
            ValueError -- wrong rType
        """
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
            return (pas, pos, ite)
        else:
            raise ValueError("Unknown node computation type.")

    def computeMuImpact(self, mus=[5, 10, 50, 100, 500, 1000, 5000], error=0.1):
        """Compute the impact of Mu

        The model is NOT LINEAR in mu. It is hard to guess what would be the effect
        of an increased suceptibility and it may highly modify the response of the coil.
        However, for some configuration the error is low (typically a long coil and a small projectile).

        This function provides some help to know if we can consider the model linear in Mu.
        Simply provide a range of possible susceptibilities for your projectile, and an
        acceptable relative error.

        We do not check the whole linearity, but simply in two points selected empirically.
        Therefore some care should be taken regarding the output of this helper method.

        Keyword Arguments:
            mus {list} -- [description] (default: {[5, 10, 50, 100, 500, 1000, 5000]})
            error {number} -- [description] (default: {0.1})
        """
        _mu = self.mu
        res = []
        test_res = []
        print("Coil " + self._seed + " mus")
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

            femm.mi_movetranslate2(0, self.Lb / 4, 4)
            femm.mi_analyze()
            femm.mi_loadsolution()
            femm.mo_groupselectblock(1)
            test_res.append(femm.mo_getcircuitproperties("Bobine")[2] / self._i0)
        self.mu = _mu
        success = True
        errors = []
        for i in range(0, len(test_res)):
            errors.append(numpy.abs((res[i] / res[0]) / (test_res[i] / test_res[0]) - 1))
            if errors[-1] > error:
                success = False
                # break
        if success:
            return {'valid': True,
                    'mus': mus,
                    'mu_Lz_0': res,
                    'mu_Lz_1': test_res,
                    'errors': errors}
        else:
            return {'valid': False,
                    'mus': mus,
                    'mu_Lz_0': res,
                    'mu_Lz_1': test_res,
                    'errors': errors}

    def estFreq(self):
        """Estimated frequency of variation dLz

        Based on some simple assumptions regarding the problem
        definition. This is very empirical and far from perfect for weird configurations.

        Returns:
            number -- estimated frequency
        """
        return 4 / (min(self.Lb, self.Lp) * 10**-3)

    def __del__(self):
        """  clean temporary files on exit """
        os.remove("temp/temp" + self._seed + ".fem")
        os.remove("temp/temp" + self._seed + ".ans")
