def progbar(curr, total, full_progbar, mot=""):
    frac = curr / total
    filled_progbar = round(frac * full_progbar)
    print(mot, ':' * filled_progbar + '-' * (full_progbar - filled_progbar), '[{:>7.2%}]'.format(frac), end='\n')
