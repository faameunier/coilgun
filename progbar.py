def progbar(curr, total, full_progbar):
    frac = curr / total
    filled_progbar = round(frac * full_progbar)
    print(':' * filled_progbar + '-' * (full_progbar - filled_progbar), '[{:>7.2%}]'.format(frac), end='\n')
