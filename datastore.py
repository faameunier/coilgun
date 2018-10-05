import pandas as pd
import numpy
import atexit
from shutil import copyfile

# ==== METHODS


def populate_coils(coils):
    lp = 20
    rp = 4
    lb = numpy.linspace(20, 100, 81)
    rbi = 5
    rbo = numpy.linspace(6, 15, 10)
    temp_id = 0
    for l in lb:
        for rb in rbo:
            coil = pd.Series([temp_id, lp, rp, l, rbi, rb, 100, False, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan, numpy.nan],
                             index=['id', 'Lp', 'Rp', 'Lb', 'Rbi', 'Rbo', 'mu', 'mu_approx_valid', 'resistance', 'n_points', 'dLz_z', 'dLz', 'L0', 'splinify'])
            coils = coils.append(coil, ignore_index=True)
            temp_id += 1
    return coils


def save_all():
    store.put('coils', coils)


def update_coil(coil):
    coils.loc[coil.name] = coil


def backup():
    copyfile('store.h5', 'store_backup.h5')

# ==== MAIN


store = pd.HDFStore('store.h5')


if '/coils' not in store.keys():
    print("store empty")
    dtypes = {
        'id': 'int64',
        'Lp': float,
        'Rp': float,
        'Lb': float,
        'Rbi': float,
        'Rbo': float,
        'mu': float,
        'resistance': float,
        'n_points': float,
        'dLz_z': object,
        'dLz': object,
        'L0': float,
        'mu_points': object,
        'mu_dLz_0': object,
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
        'mu_approx_valid': [],
        'resistance': [],
        'n_points': [],
        'dLz_z': [],
        'dLz': [],
        'L0': [],
        'mu_points': [],
        'mu_dLz_0': [],
        'mu_approx_valid': bool,
    })

    coils = populate_coils(coils)
    for col, dtype in dtypes.items():
        coils[col] = coils[col].astype(dtype)
    coils.set_index(['id'], inplace=True)
    store.put('coils', coils)


coils = store['coils']

# ==== EXIT
atexit.register(backup)
atexit.register(store.close)
atexit.register(save_all)
