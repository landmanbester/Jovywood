import click
from jove.main import cli
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('jove')
log = pyscilog.get_logger('DSPEC')

@cli.command()
@click.option("-d", "--data", type=str, required=True,
              help="Path to data.fits")
@click.option("-w", "--weight", type=str, required=True,
              help="Path to weights.fits")
@click.option("-o", "--outfile", type=str, required=True,
              help='Base name of output file.')
@click.option('-nthreads', '--nthreads', type=int, default=64,
              help='Number of dask threads.')
def dspec(**kw):
    '''
    dynamic spectra and lightcurve maker
    '''
    args = OmegaConf.create(kw)
    OmegaConf.set_struct(args, True)
    pyscilog.log_to_file(args.outfile + '.log')
    pyscilog.enable_memory_logging(level=3)

    print("Input options :", file=log)
    for key in args.keys():
        print('     %25s = %s' % (key, args[key]), file=log)

    import os
    os.environ["OMP_NUM_THREADS"] = str(1)
    os.environ["OPENBLAS_NUM_THREADS"] = str(1)
    os.environ["MKL_NUM_THREADS"] = str(1)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(1)
    os.environ["NUMBA_NUM_THREADS"] = str(1)
    import numpy as np
    import xarray as xr
    from astropy.io import fits
    from pfb.utils.fits import save_fits, load_fits
    from jove.utils import madmask, SingleDomain
    import nifty7 as ift
    ift.fft.set_nthreads(args.nthreads)

    try:
        from mpi4py import MPI

        master = MPI.COMM_WORLD.Get_rank() == 0
        comm = MPI.COMM_WORLD
        ntask = comm.Get_size()
        rank = comm.Get_rank()
        master = rank == 0
        mpi = ntask > 1
    except ImportError:
        master = True
        mpi = False
        comm = None
        rank = 0

    wgtc = fits.getdata(args.weight)
    datac = fits.getdata(args.data)

    # scale weights
    for c in range(4):
        int_weight = 1.0/np.var(datac[c])
        mean_weight = np.mean(wgtc[c])
        wgtc[c] *= 10*int_weight/mean_weight

    # Set up signal domain
    nv, nt = datac[0].shape
    position_space = ift.RGSpace([nv, nt])
    sp1 = ift.RGSpace(nv)
    sp2 = ift.RGSpace(nt)

    # Set up signal model
    cfmaker = ift.CorrelatedFieldMaker('')
    cfmaker.add_fluctuations(sp1, (0.0001, 1e-4), (1, .2), (.01, .1), (-3, 1.),
                             'amp1')
    cfmaker.add_fluctuations(sp2, (0.0001, 1e-4), (1, .2), (.01, .1), (-3, 1),
                             'amp2')
    cfmaker.set_amplitude_total_offset(0., (1e-2, 1e-4))
    correlated_field = cfmaker.finalize()

    normalized_amp = cfmaker.get_normalized_amplitudes()
    pspec1 = normalized_amp[0]**2
    pspec2 = normalized_amp[1]**2
    DC = SingleDomain(correlated_field.target, position_space)

    # no non-linearity so signal is correlated field
    signal = DC @ correlated_field

    # we mask regions with zero weights
    mask = madmask(datac, wgtc)
    Mask = ift.MaskOperator(ift.makeField(signal.target, mask))

    # measurement operator
    R = Mask

    data_space = R.target
    xmean = np.zeros_like(datac)
    xstd = np.zeros_like(datac)
    for c in range(4):
        wgt = wgtc[c, ~mask]
        data = datac[c, ~mask]

        var = 1.0/wgt
        N = ift.DiagonalOperator(ift.makeField(data_space, var))

        data = ift.makeField(data_space, data[wgt > 0])

        signal_response = R(signal)

        # Minimization parameters
        ic_sampling = ift.AbsDeltaEnergyController(name="Sampling",
                deltaE=1e-3, convergence_level=2, iteration_limit=1000)
        ic_newton = ift.AbsDeltaEnergyController(name='Newton', deltaE=1e-3,
                convergence_level=2, iteration_limit=25)
        # ic_sampling.enable_logging()
        # ic_newton.enable_logging()
        minimizer = ift.NewtonCG(ic_newton)

        # initial guess
        mean = 0.1 * ift.from_random(signal_response.domain, 'normal')

        # Set up likelihood energy and information Hamiltonian
        likelihood_energy = (ift.GaussianEnergy(mean=data, inverse_covariance=N.inverse) @
                            signal_response)
        H = ift.StandardHamiltonian(likelihood_energy, ic_sampling)

        # Draw new samples to approximate the KL six times
        N_samples = 6
        for j in range(6):
            if j==5:
                # more samples in last step for better stats
                N_samples = 4*N_samples
            # Draw new samples and minimize KL
            KL = ift.MetricGaussianKL(mean, H, N_samples, True, comm=comm)
            KL, convergence = minimizer(KL)
            mean = KL.position
            ift.extra.minisanity(data, lambda x: N.inverse, signal_response,
                                 KL.position, KL.samples)

        # round of robust reweights here?

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

        xmean[c] = sc.mean.val
        xstd[c] = np.sqrt(sc.var.val)

    if master:
        datac[:, mask] = np.nan
        hdr = fits.getheader(args.data)
        hdu = fits.PrimaryHDU(header=hdr)
        hdu.data = datac
        hdu.writeto(args.data + '.data.fits', overwrite=True)
        hdu.data = xmean
        hdu.writeto(args.data + '.mean.fits', overwrite=True)
        hdu.data = xstd
        hdu.writeto(args.data + '.std.fits', overwrite=True)
