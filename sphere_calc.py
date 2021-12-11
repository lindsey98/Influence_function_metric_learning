import numpy as np
import scipy.special as sc
import matplotlib.pyplot as plt

def spherical_cap(h, sz_embedding=2048):
    I_xab = sc.betainc((sz_embedding-1)/2, 0.5, 2*h - h**2)
    ratio = 0.5 * I_xab
    return ratio

def birthday_prob(p, n):
    # no collision between any two person
    cum = 1.
    for i in range(1, n):
        cum *= (1 - i*p)
    return cum

if __name__ == '__main__':
    # h_grids = np.linspace(0, 1, 1000)
    # ratio_grid = []
    # for h in h_grids:
    #     ratio = spherical_cap(h)
    #     print("{:.100f}".format(ratio))
    #     ratio_grid.append(ratio)
    #
    # plt.plot(h_grids, ratio_grid)
    # plt.show()

    h = 0.8
    ratio = spherical_cap(h)
    print("{:.100f}".format(ratio))
    prob_nocollision = birthday_prob(ratio, 11318)
    print("{:.100f}".format(prob_nocollision))
