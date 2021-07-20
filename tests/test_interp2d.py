"""
Converts nifty mf demo into time/freq interpolation script

"""

import sys

import numpy as np
import matplotlib.pyplot as plt

import nifty7 as ift

"""
g_ML, JHJ

g = exp(amp) exp(i phi)

g_ML = S g + eps, where eps ~ N(0, JHJinv)

R = S g
"""


# class MaskOperator(ift.LinearOperator):
#     """Implementation of a mask response

#     Takes a field, applies flags and returns the values of the field in a
#     :class:`UnstructuredDomain`.

#     Parameters
#     ----------
#     flags : Field
#         Is converted to boolean. Where True, the input field is flagged.
#     """
#     def __init__(self, flags):
#         if not isinstance(flags, Field):
#             raise TypeError
#         self._domain = DomainTuple.make(flags.domain)
#         self._flags = np.logical_not(flags.val)
#         self._target = DomainTuple.make(UnstructuredDomain(self._flags.sum()))
#         self._capability = self.TIMES | self.ADJOINT_TIMES

#     def apply(self, x, mode):
#         self._check_input(x, mode)
#         x = x.val
#         if mode == self.TIMES:
#             res = x[self._flags]
#             return Field(self.target, res)
#         res = np.empty(self.domain.shape, x.dtype)
#         res[self._flags] = x
#         res[~self._flags] = 0
#         return Field(self.domain, res)


def madmask(image):
    import scipy
    sig = scipy.stats.median_abs_deviation(image[image!=0], scale='normal')
    mask = (np.abs(image) > 3*sig)
    sigt = 11
    signu = 11
    from scipy.signal import convolve2d
    cmask = convolve2d(mask, np.ones((sigt, signu), dtype=np.float32), mode="same")
    return (np.abs(cmask) > 0.1)


def main():
    # Preparing the filename string for store results
    basename = "/home/landman/Data/MeerKAT/Jove/dspec/"

    from astropy.io import fits
    wgt = fits.getdata(basename + 'Weights.fits') #[:, :, 0:2065]  # select Stokes I as example
    data = fits.getdata(basename + 'TARGET/1608538564_20:09:36.999_-20:26:47.350.fits') #[:, :, 0:2065]

     # get madmask
    mask = np.ones(wgt[0].shape, dtype=bool)
    for i in range(4):
        mask = np.logical_and(mask, madmask(data[i]))

    wgt = wgt[0] * (~mask)
    data = data[0] * (~mask)

    # data = np.where(wgt > 0, data/wgt, 0.0)
    # print(np.sum(np.abs(data)>0))

    plt.figure('1')
    plt.imshow(np.log(1 + data), cmap='RdBu', vmin=-0.01, vmax=0.01, origin='lower')
    plt.colorbar()
    plt.show()

    quit()

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

    # print(S._ldiag.shape)

    # import matplotlib.pyplot as plt
    # plt.plot(np.log(S._ldiag[0]))
    # plt.show()

    # quit()

    # measurement operator
    R = Mask @ HT

    data_space = R.target

    # set variances
    var = 1.0/wgt[wgt>0]
    # var = np.where(wgt>0, 1.0, 0)
    N = ift.DiagonalOperator(ift.makeField(data_space, var))

    # # Create mock data
    # MOCK_SIGNAL = S.draw_sample_with_dtype(dtype=np.float64)
    # MOCK_NOISE = N.draw_sample_with_dtype(dtype=np.float64)
    # data = R(MOCK_SIGNAL) + MOCK_NOISE

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

def myinterp2d():
    # Preparing the filename string for store results
    basename = "/home/landman/Data/MeerKAT/Jove/dspec/"

    from astropy.io import fits
    wgt = fits.getdata(basename + 'Weights.fits')[:, :, 0:2065]  # select Stokes I as example
    data = fits.getdata(basename + 'TARGET/1608538564_20:09:36.999_-20:26:47.350.fits')[:, :, 0:2065]

     # get madmask
    mask = np.ones(wgt[0].shape, dtype=bool)
    print(mask.shape, data.shape)
    for i in range(4):
        mask = np.logical_and(mask, madmask(data[i]))

    wgt = wgt[0] * (~mask)
    data = data[0] * (~mask)

    # plt.imshow(data, cmap='RdBu', vmin=-0.01, vmax=0.01)
    # plt.colorbar()
    # plt.show()

    # quit()

    nchan, ntime = wgt.shape

    nu = np.linspace(0, 1, nchan)
    t = np.linspace(0, 1, ntime)

    from africanus.gps.utils import abs_diff

    tt = abs_diff(t, t)
    lt = 25/ntime
    sigma_f = 0.1
    Kt = sigma_f**2 * np.exp(-tt**2/(2*lt**2))
    vv = abs_diff(nu, nu)
    lv = 25/nchan
    Kv = np.exp(-vv**2/(2*lv**2))
    K = (Kv, Kt)
    from pfb.utils.misc import kron_matvec
    def Ky(x):
        return kron_matvec(K, x).reshape(nchan, ntime) + wgt * x

    from pfb.opt.pcg import pcg
    delta = pcg(Ky, data, np.zeros((nchan, ntime), dtype=np.float64), minit=10,
                verbosity=2, report_freq=1, tol=5e-4)
    sol = kron_matvec(K, delta).reshape(nchan, ntime)

    plt.figure('1')
    plt.imshow(sol, cmap='RdBu', origin='lower') #, vmin=-0.01, vmax=-0.01)
    plt.colorbar()

    plt.show()



if __name__ == '__main__':
    main()
    # myinterp2d()
