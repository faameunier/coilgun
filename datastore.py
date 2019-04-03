import pandas as pd
import numpy
import atexit
from shutil import copyfile

# ==== METHODS


def populate_coils(coils):
    """Populate the dataframe with values to study

    Set all coils geometry that need to be studied.
    Here values are fixed for my particular research.

    Arguments:
        coils {DataFrame} -- Empty coil dataframe

    Returns:
        DataFrame -- populated df
    """
    lp = 20
    rp = 4
    lb = numpy.linspace(20, 100, 81)
    rbi = 5
    rbo = numpy.linspace(6, 11, 6)
    temp_id = 0
    phi = 1
    for l in lb:
        for rb in rbo:
            coil = pd.Series([temp_id, lp, rp, l, rbi, rb, 100, False, phi, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan],
                             index=['id', 'Lp', 'Rp', 'Lb', 'Rbi', 'Rbo', 'mu', 'mu_approx_valid', 'phi', 'resistance', 'n_points', 'dLz_z', 'dLz', 'L0', 'mu_points', 'mu_Lz_0'])
            coils = coils.append(coil, ignore_index=True)
            temp_id += 1
    return coils


def populate_setups(setups):
    """Populatte setups

    Set all electrical setups
    that will be tested.

    Here again these are the ones
    I wanted to explore.

    Arguments:
        setups {DataFrame} -- Empty setups dataframe

    Returns:
        DataFrame -- populated df
    """
    E = 400
    C = 0.0024
    R = 0.07
    setup = pd.Series([0, C, E, R],
                      index=['id', 'C', 'E', 'R'])
    setups = setups.append(setup, ignore_index=True)
    setup = pd.Series([0, C, E / 2, R],
                      index=['id', 'C', 'E', 'R'])
    setups = setups.append(setup, ignore_index=True)
    setup = pd.Series([0, C / 2, E, R],
                      index=['id', 'C', 'E', 'R'])
    setups = setups.append(setup, ignore_index=True)
    return setups


def save_all():
    """Save all stores"""
    store.put('coils', coils)
    store.put('setups', setups)
    store.put('solutions', solutions)


def update_coil(coil):
    """update a given coil"""
    coils.loc[coil.name] = coil


def save_setup(setup):
    """save setupd df"""
    global setups
    setups = setups.append(setup, ignore_index=True)


def save_solution(solution):
    """save solution df"""
    global solutions
    solutions = solutions.append(solution, ignore_index=True)


def backup():
    """create a backup of the database in case of problem"""
    copyfile('store.h5', 'store_backup.h5')

# ==== MAIN


# read HDFstore
store = pd.HDFStore('store.h5')


# ==== COILS CHECK
if '/coils' not in store.keys():
    print("coils store empty")
    dtypes = {
        'id': 'int64',
        'Lp': float,
        'Rp': float,
        'Lb': float,
        'Rbi': float,
        'Rbo': float,
        'mu': float,
        'phi': float,
        'resistance': float,
        'n_points': float,
        'dLz_z': object,
        'dLz': object,
        'L0': float,
        'mu_points': object,
        'mu_Lz_0': object,
        'mu_approx_valid': bool,
    }

    coils = pd.DataFrame({
        'id': [],
        'Lp': [],
        'Rp': [],
        'Lb': [],
        'Rbi': [],
        'Rbo': [],
        'mu': [],
        'phi': [],
        'mu_approx_valid': [],
        'resistance': [],
        'n_points': [],
        'dLz_z': [],
        'dLz': [],
        'L0': [],
        'mu_points': [],
        'mu_Lz_0': [],
        'mu_approx_valid': bool,
    })

    coils = populate_coils(coils)
    for col, dtype in dtypes.items():
        coils[col] = coils[col].astype(dtype)
    coils.set_index(['id'], inplace=True)
    store.put('coils', coils)


coils = store['coils']

# ==== SETUP CHECK
if '/setups' not in store.keys():
    print("setups store empty")
    dtypes = {
        'id': 'int64',
        'C': float,
        'E': float,
        'R': float,
    }

    setups = pd.DataFrame({
        'id': [],
        'C': [],
        'E': [],
        'R': [],
    })

    setups = populate_setups(setups)
    for setup, dtype in dtypes.items():
        setups[setup] = setups[setup].astype(dtype)
    setups.set_index(['id'], inplace=True)
    store.put('setups', setups)


setups = store['setups']

# ==== SOLUTION CHECK
if '/solutions' not in store.keys():
    print("solutions store empty")
    dtypes = {
        'id': 'int64',
        'coil': 'int64',
        'setup': 'int64',
        'z0': float,
        'v0': float,
        'v1': float,
        'Ec': float,
        'tau': float,
        'chained': 'int64',
    }

    solutions = pd.DataFrame({
        'id': [],
        'coil': [],
        'setup': [],
        'z0': [],
        'v0': [],
        'v1': [],
        'Ec': [],
        'tau': [],
        'chained': [],
    })

    solutions.set_index(['id'], inplace=True)
    store.put('solutions', solutions)


solutions = store['solutions']

# ==== EXIT
# Make sure everything is saved and properly closed. We do not want any errors here.
atexit.register(backup)
atexit.register(store.close)
atexit.register(save_all)
