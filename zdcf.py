import numpy as np
import pandas as pd
from astropy.time import Time
import itertools
from scipy.stats.stats import pearsonr
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator

'''
ZDCF parameters
'''
N_MIN = 11  # minimum number of data points in one ZDCF bin
EPS = 3  # epsilon ('small parameter')
MC_LIM = 300  # number of Monte Carlo steps (if MC_LIM == 0 then skip the procedure)

'''
Plotting parameters
'''
rcParams['figure.figsize'] = 16, 9
rcParams.update({'font.size': 26})
rcParams['xtick.major.pad'] = 12
rcParams['ytick.major.pad'] = 12


def mc_err(a, b, a_err, b_err):
    """
    Monte Carlo simulations for measurement errors accounting
    """
    zeta = 0
    for i in range(MC_LIM):
        r = pearsonr(np.random.normal(a, a_err), np.random.normal(b, b_err))[0]
        zeta += np.arctanh(r)

    zeta /= MC_LIM
    return np.tanh(zeta), zeta


def zdcf(a, b, max_shift=1, show_plot=True, title='ZDCF'):
    """
    Z-transformed discrete correlation function. For more details see paper by Prof. Tal Alexander:
    'Improved AGN light curve analysis with the z-transformed discrete correlation function' (2013).

    :param a: first light curve (pandas type with 'epoch', 'flux' and 'err' properties)
    :param b: second light curve (pandas type with 'epoch', 'flux' and 'err' properties)
    :param max_shift: limit on the maximum (and minimum) time lag
    :param show_plot: whether or not to display a plot 'corr. coefficient vs time lag'
    :param title: the title of the displayed plot

    :return: description TBU
    """

    # arrays for storing the result
    tau, corr = ([] for i in range(2))
    tau_edg, corr_err = ([[], []] for i in range(2))

    # array of flux values and their indexes
    pairs = {'fluxes': np.array(list(itertools.product(a.flux, b.flux))),
             'errors': np.array(list(itertools.product(a.err, b.err))),
             'indexes': np.array(list(itertools.product(range(a.shape[0]), range(b.shape[0]))))}

    # calculate and sort time lags
    lags_ = list(itertools.product(a.epoch, b.epoch))
    for i, epoch in enumerate(lags_):
        lags_[i] = epoch[1] - epoch[0]
    pairs['lags'] = np.array(lags_)
    p = (pairs['lags']).argsort()

    # apply permutation
    for key in pairs:
        pairs[key] = pairs[key][p]

    # apply boundary conditions to time lags
    checkpoints = np.where(abs(pairs['lags']) < max_shift)[0]
    if len(checkpoints) > 0:
        for key in pairs:
            pairs[key] = pairs[key][checkpoints[0]:(checkpoints[-1] + 1)]
        unique_ind = min(len(np.unique(pairs['indexes'][:, 0])), len(np.unique(pairs['indexes'][:, 1])))
    else:
        unique_ind = 0

    # check if there is enough data points
    if unique_ind < N_MIN:
        return ([],) * 4

    # initialize the place in time lags array from where the binning algorithm starts (middle by default)
    dens = len(pairs['lags']) // 2

    empty = True
    bin_ = {}
    length = len(pairs['lags'])

    for s in [1, 2]:
        i = dens - (s % 2)

        while -1 < i < length:
            if empty:
                for key in pairs:
                    bin_[key] = np.array([pairs[key][i]])
                empty = False
            else:
                if not any(pairs['indexes'][i][j] in bin_['indexes'][:, j] for j in [0, 1]):
                    for key in bin_:
                        bin_[key] = np.append(bin_[key], [pairs[key][i]], axis=0)

            if (len(bin_['lags']) >= N_MIN) and \
                    ((i == 0) or (i == length - 1) or (abs(pairs['lags'][i + (-1) ** s] - bin_['lags'][-1]) > EPS)):
                tau_av = np.average(bin_['lags'])
                tau += [tau_av]
                if s == 2:
                    tau_edg[0] += [tau_av - bin_['lags'][0]]
                    tau_edg[1] += [bin_['lags'][-1] - tau_av]
                else:
                    tau_edg[0] += [tau_av - bin_['lags'][-1]]
                    tau_edg[1] += [bin_['lags'][0] - tau_av]

                a, b = zip(*bin_['fluxes'])
                a_err, b_err = zip(*bin_['errors'])

                # z-transformation and Monte Carlo procedure
                if MC_LIM > 0:
                    r, zeta = mc_err(a, b, a_err, b_err)
                else:
                    r = pearsonr(a, b)[0]
                    zeta = np.arctanh(r)
                corr += [r]

                n_1 = len(bin_['lags']) - 1

                # Tal Alexander (6), ansatz rho = r is used
                z_av = zeta + r / (2 * n_1) * \
                       (1 + (5 + r ** 2) / (4 * n_1) + (11 + 2 * r ** 2 + 3 * r ** 4) / (8 * n_1 ** 2))

                # Tal Alexander (7)
                s_z2 = 1 / n_1 * (1 + (4 - r ** 2) / (2 * n_1) + (22 - 6 * r ** 2 - 3 * r ** 4) / (6 * n_1 ** 2))
                s_z = np.sqrt(s_z2)

                corr_err[0] += [abs(np.tanh(z_av - s_z) - r)]
                corr_err[1] += [abs(np.tanh(z_av + s_z) - r)]

                empty = True

            # move in positive or negative direction (towards bounds)
            i += (-1) ** s

        empty = True

    if len(tau) == 0:
        return ([],) * 4

    tau, tau_edg = np.array(tau), np.array(tau_edg)
    corr, corr_err = np.array(corr), np.array(corr_err)

    p = tau.argsort()
    tau, corr = tau[p], corr[p]
    for i in range(2):
        tau_edg[i], corr_err[i] = tau_edg[i][p], corr_err[i][p]

    # draw plot if necessary
    if show_plot:
        fig, ax = plt.subplots()

        plt.errorbar(tau, corr, xerr=tau_edg, yerr=corr_err, fmt='.')
        plt.title(title)
        ax.set_xlabel('Time delay (days)')
        ax.set_ylabel('Correlation coefficient')

        plt.xlim(-max_shift, max_shift)
        plt.ylim(-1.0, 1.0)

        ax.xaxis.set_major_locator(MultipleLocator(max_shift / 5))
        ax.xaxis.set_minor_locator(MultipleLocator(max_shift / 10))

        ax.tick_params(direction='in', which='both')
        ax.tick_params(length=12, width=3, which='major')
        ax.tick_params(length=6, width=3, which='minor')
        ax.grid(True, which='both', linestyle='dashed')
        plt.show()
        plt.close()

    return tau, tau_edg, corr, corr_err


if __name__ == '__main__':
    # read the weekly binned gamma-ray light curve of the quasar 4FGL J1512.8-0906 (1510-089)
    gamma = pd.read_csv('J1512.8-0906',
                        names=['source', 'start', 'end', 'flux', 'err'],
                        dtype={('start', 'end', 'flux', 'err'): np.float16},
                        delimiter=',',
                        comment='#')
    gamma['epoch'] = (gamma.start + gamma.end) / 2

    # read the radio light curve of the quasar 4FGL J1512.8-0906 (1510-089)
    radio = pd.read_csv('1510-089',
                        names=['source', 'epoch', 'flux'],
                        dtype={'flux': np.float16},
                        parse_dates=[1],
                        usecols=[0, 1, 2],
                        delim_whitespace=True,
                        comment='#')
    radio.loc[:, 'epoch'] = Time(radio.epoch).mjd
    radio['err'] = 0.05 * radio['flux']

    # perform ZDCF correlation analysis
    zdcf(gamma, radio, 500, True, r'4FGL J1512.8$-$0906 (1510$-$089) ZDCF')
