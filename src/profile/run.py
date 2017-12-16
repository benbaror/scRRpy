from scrrpy import DRR

if __name__ == '__main__':
    drr = DRR(0.1)
    Drr, Drr_err = drr.drr(1, tol=0.0, neval=1e2)
