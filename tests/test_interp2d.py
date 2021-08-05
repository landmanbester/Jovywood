"""
Converts nifty mf demo into time/freq interpolation script

"""

import sys

import numpy as np
import matplotlib as mpl
mpl.rcParams.update({'font.size': 8, 'font.family': 'serif'})
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import nifty7 as ift
from astropy.io import fits
from pfb.operators.mask import Mask
from africanus.gps.utils import abs_diff
from pfb.utils.misc import kron_matvec as kv
from pfb.opt.pcg import pcg
from jove.utils import madmask, SingleDomain


# def sample(xihat, R, L, Sigma, hess):
#     eps = np.sqrt(Sigma) * np.random.randn(xihat.shape)
#     y = R(xi) + eps
#     b = R.hdot(y/Sigma)
#     xhat = pcg(hess, b, np.zeros_like(b),
#                minit=10, maxit=100, verbosity=2,
#                report_freq=1, tol=1e-2)



def singlespec():
    # Preparing the filename string for store results
    basename = "/home/landman/Data/MeerKAT/Jove/dspec/"

    from astropy.io import fits
    wgt = fits.getdata(basename + 'Weights.fits') #[:, :, 0:2065]  # select Stokes I as example
    data = fits.getdata(basename + 'TARGET/1608538564_20:09:36.999_-20:26:47.350.fits') #[:, :, 0:2065]

     # refine mask
    mask = madmask(data, wgt)
    wgt = wgt[0] * (~mask)
    data = data[0] * (~mask)

    # plt.figure('1')
    # plt.imshow(np.log(1 + data), cmap='RdBu', vmin=-0.01, vmax=0.01, origin='lower')
    # plt.colorbar()
    # plt.show()

    # Set up signal domain
    npix1, npix2 = wgt.shape
    position_space = ift.RGSpace([npix1, npix2])

    # Define harmonic space and harmonic transform
    hspace = position_space.get_default_codomain()
    HT = ift.HarmonicTransformOperator(hspace, position_space)

    # Domain on which the field's degrees of freedom are defined
    domain = ift.DomainTuple.make(hspace)

    # we mask regions with zero weights
    mask = np.where(wgt > 0, 0.0, 1.0)
    mask = ift.Field.from_raw(position_space, mask)
    Mask = ift.MaskOperator(mask)

    # Define amplitude (square root of power spectrum)
    def power_spectrum(k):
        return 0.0001/(1. + k**2)**2

    def sqrtpspec(k):
        return np.sqrt(power_spectrum(k))

    # 1D spectral space on which the power spectrum is defined
    power_space = ift.PowerSpace(hspace)

    # Mapping to (higher dimensional) harmonic space
    PD = ift.PowerDistributor(hspace, power_space)

    # Apply the mapping
    prior_correlation_structure = PD(ift.PS_field(power_space, power_spectrum))

    # Insert the result into the diagonal of an harmonic space operator
    S = ift.DiagonalOperator(prior_correlation_structure)

    # measurement operator
    R = Mask @ HT

    data_space = R.target

    # set variances
    var = 1.0/wgt[wgt>0]
    N = ift.DiagonalOperator(ift.makeField(data_space, var))
    data = ift.makeField(data_space, data[wgt > 0])


    # ##### WIENER FILTER solution #####
    # D_inv = R.adjoint @ N.inverse @ R + S.inverse
    # j = R.adjoint_times(N.inverse_times(data))
    # IC = ift.GradientNormController(iteration_limit=500, tol_abs_gradnorm=1e-1)
    # D = ift.InversionEnabler(D_inv, IC, approximation=S.inverse).inverse
    # m = D(j)

    # rg = isinstance(position_space, ift.RGSpace)
    # plot = ift.Plot()
    # filename = "/home/landman/Data/MeerKAT/Jove/dspec/wiener.png"
    # # plot.add(HT(MOCK_SIGNAL), title='Mock Signal')
    # plot.add(Mask.adjoint(data), title='Data', vmin=-0.01, vmax=0.01)
    # plot.add(HT(m), title='Reconstruction', vmin=-0.01, vmax=0.01)
    # plot.add(Mask.adjoint(data) - Mask.adjoint(Mask(HT(m))),
    #             title='Residuals', vmin=-0.01, vmax=0.01)
    # plot.output(nx=3, ny=1, xsize=10, ysize=10, name=filename)
    # print("Saved results as '{}'.".format(filename))


    # #### MAP soln + samples with known power spectrum #####
    # a = ift.PS_field(power_space, sqrtpspec)
    # A = ift.makeOp(PD(a))
    # R = Mask @ HT @ A

    # IC = ift.DeltaEnergyController(
    #     name='Newton', iteration_limit=100, tol_rel_deltaE=1e-4)
    # likelihood_energy = ift.GaussianEnergy(mean=data, inverse_covariance=N.inverse) @ R
    # Ham = ift.StandardHamiltonian(likelihood_energy, IC)
    # initial_position = ift.from_random(domain, 'normal')
    # H = ift.EnergyAdapter(initial_position, Ham, want_metric=True)

    # # Minimize
    # minimizer = ift.NewtonCG(IC)
    # H, _ = minimizer(H)

    # # Draw posterior samples
    # N_samples = 5
    # metric = Ham(ift.Linearization.make_var(H.position, want_metric=True)).metric
    # samples = [metric.draw_sample(from_inverse=True) + H.position
    #            for _ in range(N_samples)]

    # sc = ift.StatCalculator()
    # for s in samples:
    #     sc.add(HT(A(s)))

    # # Plotting
    # rg = isinstance(position_space, ift.RGSpace)
    # plot = ift.Plot()
    # filename = "/home/landman/Data/MeerKAT/Jove/dspec/target/map.png"
    # plot.add(HT(MOCK_SIGNAL), title='Mock Signal')
    # plot.add(Mask.adjoint(data), title='Data')
    # plot.add(HT(A(H.position)), title='MAP')
    # plot.add(sc.mean, title='Mean')
    # plot.add(Mask.adjoint(HT(A(H.position) - HT(MOCK_SIGNAL))),
    #             title='Residuals')
    # plot.add(ift.sqrt(sc.var), title='Uncertainty')
    # plot.add(Mask.adjoint(MOCK_NOISE), title='Noise')
    # plot.add(wgt, title='Weights')

    # plot.output(nx=3, ny=3, xsize=10, ysize=10, name=filename)
    # print("Saved results as '{}'.".format(filename))


    ##### Inference with correlated field with unknown power spectrum #####
    args = {
        'offset_mean': 0,
        'offset_std': (1e-3, 1e-6),

        # Amplitude of field fluctuations
        'fluctuations': (1., 0.8),  # 1.0, 1e-2

        # Exponent of power law power spectrum component
        'loglogavgslope': (-3., 1),  # -6.0, 1

        # Amplitude of integrated Wiener process power spectrum component
        'flexibility': (2, 1.),  # 1.0, 0.5

        # How ragged the integrated Wiener process component is
        'asperity': (0.5, 0.4)  # 0.1, 0.5
    }

    correlated_field = ift.SimpleCorrelatedField(position_space, **args)
    pspec = correlated_field.power_spectrum

    # no non-linearity so signal is correlated field
    # DC = SingleDomain(correlated_field.target, position_space)
    signal = correlated_field

    # we mask regions with zero weights
    mask = np.where(wgt > 0, 0.0, 1.0)
    mask = ift.makeField(signal.target, mask)
    Mask = ift.MaskOperator(mask)

    R = Mask

    signal_response = R(signal)

    # Minimization parameters
    ic_sampling = ift.AbsDeltaEnergyController(name="Sampling (linear)",
            deltaE=0.05, iteration_limit=100)
    ic_newton = ift.AbsDeltaEnergyController(name='Newton', deltaE=0.5,
            convergence_level=2, iteration_limit=35)
    minimizer = ift.NewtonCG(ic_newton)
    ic_sampling_nl = ift.AbsDeltaEnergyController(name='Sampling (nonlin)',
            deltaE=0.5, iteration_limit=15, convergence_level=2)
    minimizer_sampling = ift.NewtonCG(ic_sampling_nl)

    # Set up likelihood energy and information Hamiltonian
    likelihood_energy = (ift.GaussianEnergy(mean=data, inverse_covariance=N.inverse) @
                         signal_response)
    H = ift.StandardHamiltonian(likelihood_energy, ic_sampling)

    initial_mean = ift.MultiField.full(H.domain, 0.)
    mean = initial_mean

    mock_position = ift.from_random(signal_response.domain, 'normal')

    filename = "/home/landman/Data/MeerKAT/Jove/dspec/setup.png"
    plot = ift.Plot()
    plot.add(signal(mock_position), title='Initial sample', zmin = 0, zmax = 1)
    plot.add(R.adjoint_times(data), title='Data')
    plot.add([pspec.force(mock_position)], title='Power Spectrum')
    plot.output(ny=1, nx=3, xsize=24, ysize=6, name=filename)

    # number of samples used to estimate the KL
    N_samples = 10

    # Draw new samples to approximate the KL six times
    filename = "/home/landman/Data/MeerKAT/Jove/dspec/samples_{}.png"
    for i in range(6):
        if i==5:
            # Double the number of samples in the last step for better statistics
            N_samples = 2*N_samples
        # Draw new samples and minimize KL
        KL = ift.GeoMetricKL(mean, H, N_samples, minimizer_sampling, True)
        KL, convergence = minimizer(KL)
        mean = KL.position
        ift.extra.minisanity(data, lambda x: N.inverse, signal_response,
                             KL.position, KL.samples)

        # Plot current reconstruction
        plot = ift.Plot()
        plot.add(signal(KL.position), title="Latent mean", zmin = 0, zmax = 1)
        plot.add([pspec.force(KL.position + ss) for ss in KL.samples],
                 title="Samples power spectrum")
        plot.output(ny=1, ysize=6, xsize=16,
                    name=filename.format("loop_{:02d}".format(i)))

    sc = ift.StatCalculator()
    for sample in KL.samples:
        sc.add(signal(sample + KL.position))

    # Plotting
    filename_res = filename.format("results")
    plot = ift.Plot()
    plot.add(sc.mean, title="Posterior Mean", zmin = 0, zmax = 1)
    plot.add(ift.sqrt(sc.var), title="Posterior Standard Deviation")

    powers = [pspec.force(s + KL.position) for s in KL.samples]
    sc = ift.StatCalculator()
    for pp in powers:
        sc.add(pp.log())
    plot.add(
        powers + [pspec.force(mock_position),
                  pspec.force(KL.position), sc.mean.exp()],
        title="Sampled Posterior Power Spectrum",
        linewidth=[1.]*len(powers) + [3., 3., 3.],
        label=[None]*len(powers) + ['Ground truth', 'Posterior latent mean', 'Posterior mean'])
    plot.output(ny=1, nx=3, xsize=24, ysize=6, name=filename_res)
    print("Saved results as '{}'.".format(filename_res))

def inspect():
    basename = "/home/landman/Data/MeerKAT/Jove/dspec/"
    from astropy.io import fits
    wgt = fits.getdata(basename + 'Weights.fits')[:, :, 0:2065]  # select Stokes I as example
    data = fits.getdata(basename + 'TARGET/1608538564_20:09:36.999_-20:26:47.350.fits')[:, :, 0:2065]
    # refine mask
    mask = madmask(data, wgt)
    wgt = wgt[0] * (~mask)
    data = data[0] * (~mask)

    dct = np.load(basename + 'ift_result.npz')
    mean = dct['mean']
    std = np.sqrt(dct['var'])

    fig, ax = plt.subplots(nrows=2, ncols=2)

    ax[0,0].imshow(data, vmix=0.1*data.min(), vmax=0.1*data.max())
    ax[0,0].colorbar()

    ax[0,1].imshow(wgt, vmix=0.1*wgt.min(), vmax=0.1*wgt.max())
    ax[0,1].colorbar()

    ax[1,0].imshow(mean, vmix=0.1*mean.min(), vmax=0.1*mean.max())
    ax[1,0].colorbar()

    ax[1,1].imshow(std, vmix=0.1*std.min(), vmax=0.1*std.max())
    ax[1,1].colorbar()

    plt.show()

def main():
    # Preparing the filename string for store results
    basename = "/home/landman/Data/MeerKAT/Jove/dspec/"

    from astropy.io import fits
    wgt = fits.getdata(basename + 'Weights.fits')[:, :, 0:2065]  # select Stokes I as example
    data = fits.getdata(basename + 'TARGET/1608538564_20:09:36.999_-20:26:47.350.fits')[:, :, 0:2065]

    # refine mask
    mask = madmask(data, wgt)
    wgt = 100 * wgt[0] * (~mask)
    data = data[0] * (~mask)

    # import matplotlib.pyplot as plt
    # tmp = [mask, wgt, data]
    # for im in tmp:
    #     plt.figure()
    #     plt.imshow(im, vmin=im.min(), vmax=0.1*im.max())
    #     plt.colorbar()
    #     plt.show()

    # quit()

    # Set up signal domain
    npix1, npix2 = wgt.shape
    position_space = ift.RGSpace([npix1, npix2])
    sp1 = ift.RGSpace(npix1)
    sp2 = ift.RGSpace(npix2)

    # Set up signal model
    cfmaker = ift.CorrelatedFieldMaker('')
    cfmaker.add_fluctuations(sp1, (0.005, 1e-2), (1, .2), (.01, .5), (-3, 1.),
                             'amp1')
    cfmaker.add_fluctuations(sp2, (0.005, 1e-2), (1, .2), (.01, .5), (-3, 1),
                             'amp2')
    cfmaker.set_amplitude_total_offset(0., (1e-2, 1e-4))
    # cfmaker.add_fluctuations_matern(sp1, (1e-4, 1e-2), (10, 1), (-5, .5), 'amp1') #, adjust_for_volume=False)
    # cfmaker.add_fluctuations_matern(sp2, (1e-4, 1e-2), (10, 1), (-10, .5), 'amp2') #, adjust_for_volume=False)
    # cfmaker.set_amplitude_total_offset(0, (0.05, 0.05))
    correlated_field = cfmaker.finalize()

    normalized_amp = cfmaker.get_normalized_amplitudes()
    pspec1 = normalized_amp[0]**2
    pspec2 = normalized_amp[1]**2
    DC = SingleDomain(correlated_field.target, position_space)

    # no non-linearity so signal is correlated field
    signal = DC @ correlated_field

    # we mask regions with zero weights
    mask = np.where(wgt > 0, 0.0, 1.0)
    mask = ift.makeField(signal.target, mask)
    Mask = ift.MaskOperator(mask)

    # measurement operator
    R = Mask

    data_space = R.target

    # set variances
    var = 1.0/wgt[wgt>0]
    N = ift.DiagonalOperator(ift.makeField(data_space, var))

    data = ift.makeField(data_space, data[wgt > 0])

    signal_response = R(signal)

    # Minimization parameters
    ic_sampling = ift.AbsDeltaEnergyController(name="Sampling",
            deltaE=0.01, iteration_limit=1000)
    ic_newton = ift.AbsDeltaEnergyController(name='Newton', deltaE=0.01,
            convergence_level=2, iteration_limit=25)
    ic_sampling.enable_logging()
    ic_newton.enable_logging()
    minimizer = ift.NewtonCG(ic_newton)

    # ic_sampling_nl = ift.AbsDeltaEnergyController(name='Sampling (nonlin)',
    #         deltaE=0.5, iteration_limit=15, convergence_level=2)
    # minimizer_sampling = ift.NewtonCG(ic_sampling)

    # Set up likelihood energy and information Hamiltonian
    likelihood_energy = (ift.GaussianEnergy(mean=data, inverse_covariance=N.inverse) @
                         signal_response)
    H = ift.StandardHamiltonian(likelihood_energy, ic_sampling)

    # initial_mean = ift.MultiField.full(H.domain, 0.)
    initial_mean = 0.1 * ift.from_random(signal_response.domain, 'normal')
    mean = initial_mean

    filename = "/home/landman/Data/MeerKAT/Jove/dspec/setup.png"
    plot = ift.Plot()
    # for i in range(N_samples):
    samp = ift.from_random(signal_response.domain, 'normal')
    plot.add(signal(samp), title='Initial sample') # , vmin=-0.1, vmax=0.1)
    plot.add(R.adjoint_times(data), title='Data', vmin=-0.01, vmax=0.01)
    plot.add([pspec1.force(samp)], title='Power Spectrum 1')
    plot.add([pspec2.force(samp)], title='Power Spectrum 2')

    plot.output(nx=2, ny=2, xsize=16, ysize=16, name=filename)

    # number of samples used to estimate the KL
    N_samples = 5

    # Draw new samples to approximate the KL six times
    filename = "/home/landman/Data/MeerKAT/Jove/dspec/samples_{}.png"
    for i in range(6):
        if i==5:
            # Double the number of samples in the last step for better statistics
            N_samples = 2*N_samples
        # Draw new samples and minimize KL
        KL = ift.MetricGaussianKL(mean, H, N_samples, True)
        KL, convergence = minimizer(KL)
        mean = KL.position
        ift.extra.minisanity(data, lambda x: N.inverse, signal_response,
                             KL.position, KL.samples)

        # Plot current reconstruction
        plot = ift.Plot()
        plot.add(signal(KL.position), title="Latent mean", zmin = 0, zmax = 1)
        plot.add([pspec1.force(KL.position + ss) for ss in KL.samples],
                 title="Samples pspec 1")
        plot.add([pspec2.force(KL.position + ss) for ss in KL.samples],
                 title="Samples pspec 1")
        plot.output(nx=3, ny=1, ysize=6, xsize=24,
                    name=filename.format("loop_{:02d}".format(i)))

    sc = ift.StatCalculator()
    scA1 = ift.StatCalculator()
    scA2 = ift.StatCalculator()
    powers1 = []
    powers2 = []
    for sample in KL.samples:
        sc.add(signal(sample + KL.position))
        p1 = pspec1.force(sample + KL.position)
        p2 = pspec2.force(sample + KL.position)
        scA1.add(p1)
        powers1.append(p1)
        scA2.add(p2)
        powers2.append(p2)

    # Plotting
    filename_res = filename.format("results")
    plot = ift.Plot()
    plot.add(sc.mean, title="Posterior Mean", zmin = 0, zmax = 1)
    plot.add(ift.sqrt(sc.var), title="Posterior Standard Deviation")

    # powers1 = [pspec1.force(s + KL.position) for s in KL.samples]
    # powers2 = [pspec2.force(s + KL.position) for s in KL.samples]
    plot.add(powers1 + [scA1.mean, pspec1.force(initial_mean)],
             title="Sampled Posterior Power Spectrum 1",
             linewidth=[1.]*len(powers1) + [3., 3.])
    plot.add(powers2 + [scA2.mean, pspec2.force(initial_mean)],
             title="Sampled Posterior Power Spectrum 2",
             linewidth=[1.]*len(powers2) + [3., 3.])
    plot.output(ny=2, nx=2, xsize=15, ysize=15, name=filename_res)
    print("Saved results as '{}'.".format(filename_res))


    np.savez('/home/landman/Data/MeerKAT/Jove/dspec/ift_result.npz', mean=sc.mean, var=sc.var)

def sparse_rec():
    # Preparing the filename string for store results
    basename = "/home/landman/Data/MeerKAT/Jove/dspec/"

    from astropy.io import fits
    wgt = fits.getdata(basename + 'Weights.fits')[:, :, 0:2064]  # select Stokes I as example
    data = fits.getdata(basename + 'TARGET/1608538564_20:09:36.999_-20:26:47.350.fits')[:, :, 0:2064]
    nv, nt = data[0].shape

    # refine mask
    mask = madmask(data, wgt)
    wgt = wgt[0, ~mask]
    data = data[0, ~mask]

    # # scale weights
    # int_weight = 1.0/np.var(data)
    # mean_weight = np.mean(wgt)
    # wgt *= int_weight/mean_weight

    data = data[None, :]
    wgt = wgt[None, :]

    from pfb.operators.mask import Mask
    R = Mask(1, mask)
    # R = lambda x: x

    from pfb.operators.psi import PSI
    psi = PSI((1, nv, nt),
              nlevels=2,
              bases=['db1', 'db2', 'db3', 'db4'])

    def hess(alpha):
        return psi.hdot(R.hdot(wgt * R.dot(psi.dot(alpha)))) # + alpha

    from pfb.opt.pcg import pcg
    alpha0 = psi.hdot(R.hdot(wgt * data))
    # alphahat = pcg(hess, alpha0, alpha0, maxit=50, minit=10)

    from pfb.opt.power_method import power_method
    L, Lp = power_method(hess, alpha0.shape)
    print('L = ', L)

    def fprime(alpha):
        delta = data - R.dot(psi.dot(alpha))
        return np.vdot(delta, wgt * delta)/2.0, -psi.hdot(R.hdot(wgt*delta))  #/wsum

    sigma = 1e-5
    def prox(alpha):
        return np.maximum(np.abs(alpha - sigma), 0.0) * np.sign(alpha)

    from pfb.opt.fista import fista
    alpha0 = np.zeros(alpha0.shape)
    alphahat = fista(alpha0, L, fprime, prox, report_freq=10, verbosity=2)

    xhat = psi.dot(alphahat)

    fig, ax = plt.subplots(nrows=1, ncols=2)

    # im = ax[0].imshow(R.hdot(data)[0], vmin=-0.01, vmax=0.01)
    data = data[0]
    data[mask] = np.nan
    im = ax[0].imshow(data, vmin=-0.01, vmax=0.01)
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("bottom", size="10%", pad=0.5)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=1, width=1, labelsize=5, pad=0.5)

    im = ax[1].imshow(xhat[0], vmin=-0.01, vmax=0.01)
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("bottom", size="10%", pad=0.5)
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=1, width=1, labelsize=5, pad=0.5)

    plt.show()



def myinterp2d():
    # Preparing the filename string for store results
    basename = "/home/landman/Data/MeerKAT/Jove/dspec/UHF_NoSubTarget_SourceOff/UHF_NoSubTarget_SourceOff/"
    source = 'OFF/1625623568_20:05:35.530_-20:06:02.391.fits'
    wgt = fits.getdata(basename + 'Weights.fits')  # select Stokes I as example
    data = fits.getdata(basename + source)
    nv, nt = data[0].shape

    # K = Kt kron Kv
    # Ky =  Kt kron Kv + Sigmainv

    # refine mask
    th = 6
    sigv = 3
    sigt = 3
    mask = madmask(data, wgt, th=th, sigv=sigv, sigt=sigt)
    R = Mask(mask)

    nu = np.linspace(0, 1, nv)
    t = np.linspace(0, 1, nt)

    tt = abs_diff(t, t)
    lt = 0.025
    sigma_f = 0.01
    Kt = sigma_f**2 * np.exp(-tt**2/(2*lt**2))
    Lt = np.linalg.cholesky(Kt + 1e-10*np.eye(nt))
    vv = abs_diff(nu, nu)
    lv = 0.25
    Kv = np.exp(-vv**2/(2*lv**2))
    Lv = np.linalg.cholesky(Kv + 1e-10*np.eye(nv))
    K = (Kv, Kt)
    L = (Lv, Lt)
    LH = (Lv.T, Lt.T)

    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(24, 6))
    corrs = ['I', 'Q', 'U', 'V']
    for c in range(4):
        wgtc = wgt[c, ~mask]
        datac = data[c, ~mask]

        def hess(x):
            return kv(LH, R.hdot(wgtc * R.dot(kv(L, x)))) + x

        # # scale weights
        # int_weight = 1.0/np.var(data)
        # mean_weight = np.mean(wgt)
        # wgt *= int_weight/mean_weight

        # wgt *= 58
        delta = pcg(hess, kv(LH, R.hdot(wgtc * datac)),
                    np.zeros((nv, nt), dtype=np.float64),
                    minit=10, maxit=100, verbosity=2,
                    report_freq=1, tol=1e-2)
        sol = kv(L, delta)

        datac = R.hdot(datac)
        datac[mask] = np.nan
        ax[0, c].set_title(f"{corrs[c]}")

        im = ax[0, c].imshow(datac, vmin=-0.005, vmax=0.005,
                             cmap='inferno', interpolation=None)
        # ax[0, c].axis('off')

        im = ax[1, c].imshow(sol, vmin=-0.005, vmax=0.005,
                             cmap='inferno', interpolation=None)
        # ax[1, c].axis('off')
        divider = make_axes_locatable(ax[1, c])
        cax = divider.append_axes("bottom", size="10%", pad=0.5)
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=1, width=1, labelsize=5, pad=0.5)

    fig.tight_layout()
    plt.savefig(basename + source + f".th{th}_sigv{sigv}_sigt{sigt}_lv{lv}_lt{lt}.png",
                dpi=500, bbox_inches='tight')
    plt.show()



if __name__ == '__main__':
    # main()
    myinterp2d()
    # inspect()
    # sparse_rec()