import click
from jove.dspec import cli
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('jove')
log = pyscilog.get_logger('GPRS')

@cli.command()
@click.option("-bn", "--basename", type=str, required=True,
              help="Path to folder containing TARGET/data.fits")
@click.option("-th", "--mad-threshold", type=float, default=6.0,
              help="Threshold for MAD flagging")
@click.option('-sigmaf', '--sigmaf', type=float, default=0.01,
              help="Zero mode of GP prior")
@click.option('-lnu', '--lnu', type=float, default=0.25,
              help="Length scale of freq covariance function")
@click.option('-lt', '--lt', type=float, default=0.05,
              help="Length scale of time covariance function")
@click.option("-ws", "--weight-scale", type=float, default=1,
              help="Weight scaling factor")
@click.option("-o", "--outfile", type=str, required=True,
              help='Base name of output file.')
@click.option('-nthreads', '--nthreads', type=int, default=8,
              help='Number of dask threads.')
@click.option('-t0', '--t0', type=int, default=0,
              help='Starting time index.')
@click.option('-tf', '--tf', type=int, default=-1,
              help='Final time index.')
@click.option('-nu0', '--nu0', type=int, default=0,
              help='Starting freq index.')
@click.option('-nuf', '--nuf', type=int, default=-1,
              help='Final freq index.')
def gpr_smooth(**kw):
    '''
    Tool to smooth dynamic spectra with GP.

    Assuming the raw dynamic spectra are obtained by performing a weighted sum
    over baselines, we have a model of the form

    d = R x + epsilon

    where R is a mask, d the data, x the solution and epsilon ~ N(0, Winv)
    where Winv is the sum of the weights computed during the weighted sum
    (also the denominator of the weighted sum). If we place a Gaussian process prior
    on x, we have to minimise

    chi2 = (d - Rx).H W (d - Rx) + (x - xbar).H Kinv (x - xbar)

    where xbar is an assumed prior mean function and Kinv the inverse of the prior
    covariance matrix (computed from the domain of the problem after assuming a
    specific covariance function).

    We approximate xbar from the data by taking the mean of the unflagged values.
    Since it is just a scalar we can simplify things by subtracting it from the data
    in which case we ant to minimise

    chi2 = (d - Rx).H W (d - Rx) + x.H Kinv x

    Next, for numerical stability, we whiten the prior by decomposing K = L L.H using
    a Cholesky decomposition. Making the change of variable x = L xi we arrive at

    chi2 = (d - R L xi).H W (d - R L xi ) + xi.H xi

    which has a solution given by

    xi = (L.H R.H W R L + I)^{-1} L.H R.H W d

    This solution is straightforward to obtain using eg. the PCG algorithm.
    The value of the variable x is given by x = L xi.
    '''
    opts = OmegaConf.create(kw)
    if opts.basename.endswith('/'):
        opts.basename = opts.basename.rstrip('/')
    OmegaConf.set_struct(opts, True)
    pyscilog.log_to_file(opts.outfile + '.log')
    pyscilog.enable_memory_logging(level=3)

    print("Input options :", file=log)
    for key in opts.keys():
        print('     %25s = %s' % (key, opts[key]), file=log)

    import os
    os.environ["OMP_NUM_THREADS"] = str(opts.nthreads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(opts.nthreads)
    os.environ["MKL_NUM_THREADS"] = str(opts.nthreads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(opts.nthreads)
    os.environ["NUMBA_NUM_THREADS"] = str(opts.nthreads)
    import numpy as np
    from scipy.signal import convolve2d
    import xarray as xr
    from astropy.io import fits
    from jove.utils import madmask, Mask
    from africanus.gps.utils import abs_diff
    from pfb.utils.misc import kron_matvec as kv
    from pfb.opt.pcg import pcg
    from pfb.utils.fits import data_from_header
    import matplotlib as mpl
    mpl.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable



    # Preparing the filename string for store results
    # basename = "/home/landman/Data/MeerKAT/Jove/dspec/FullJupiter_NoSubTarget_SourceOff/"
    # source = 'TARGET/1608538564_20:09:36.999_-20:26:47.350.fits'
    std = fits.getdata(opts.basename.rstrip('.norm.fits') + '.std.fits')[0]
    data = fits.getdata(opts.basename)[0].astype(np.float64)
    wgt = np.where(data != 0, 1.0/std**2, 0.0)
    nv, nt = data.shape
    hdr = fits.getheader(opts.basename)
    delt = hdr['CDELT1']
    phys_time, ref_time = data_from_header(hdr, axis=1)
    phys_time *= delt/3600  # sec to hr
    phys_freq, ref_freq = data_from_header(hdr, axis=2)

    if opts.tf == -1:
        opts.tf = nt
    if opts.nuf == -1:
        opts.nuf = nv

    wgt = wgt[opts.nu0:opts.nuf, opts.t0:opts.tf]
    data = data[opts.nu0:opts.nuf, opts.t0:opts.tf]

    phys_time = phys_time[opts.t0:opts.tf]
    phys_freq = phys_freq[opts.nu0:opts.nuf]

    nt = phys_time.size
    nv = phys_freq.size

    maskp = data != 0

    # refine mask
    sigv = 3
    sigt = 3
    mask = madmask(data, wgt, th=opts.mad_threshold, sigv=sigv, sigt=sigt).astype(np.bool)
    mask = ~mask

    ysize = 16
    xsize = int(np.ceil(nt * 12/nv))
    # fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(xsize, ysize))
    # ax[0].imshow(maskp.astype(np.float64), interpolation=None,)
    # ax[1].imshow(mask.astype(np.float64), interpolation=None,)


    # plt.show()

    # plt.savefig(opts.basename +
    #             f".th{opts.mad_threshold}_mask.pdf",
    #             bbox_inches='tight')
    # plt.close(fig)

    # quit()

    meandat = np.mean(data[mask])
    print(f"Using mean of data to set meanf to {meandat}", file=log)

    from pfb.utils.misc import convolve2gaussres

    sigv = 11
    sigt = 11

    xin, yin = np.meshgrid(np.arange(-(nv/2), nv/2),
                           np.arange(-(nt/2), nt/2),
                           indexing='ij')

    data = np.where(mask, data, 0.0)
    data_convolved = convolve2gaussres(data[None, :, :], xin, yin,
                                       (sigv, sigt, 0.0),
                                       opts.nthreads,
                                       norm_kernel=True)[0]

    res = np.where(mask, data - data_convolved, 0.0)

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(xsize, ysize))
    im = ax[0].imshow(data, cmap='inferno',
                 vmin=data_convolved.min(), vmax=data_convolved.max(),
                 interpolation=None,
                 aspect='auto',
                 extent=[phys_time[0], phys_time[-1],
                         phys_freq[0], phys_freq[-1]])
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="2%", pad=0.1)
    cb = fig.colorbar(im, cax=cax, orientation="vertical")
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=1, width=1, labelsize=7, pad=0.1)

    im = ax[1].imshow(data_convolved, cmap='inferno',
                 vmin=data_convolved.min(), vmax=data_convolved.max(),
                 interpolation=None,
                 aspect='auto',
                 extent=[phys_time[0], phys_time[-1],
                         phys_freq[0], phys_freq[-1]])
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="2%", pad=0.1)
    cb = fig.colorbar(im, cax=cax, orientation="vertical")
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=1, width=1, labelsize=7, pad=0.1)

    im = ax[2].imshow(res, cmap='inferno',
                 vmin=data_convolved.min(), vmax=data_convolved.max(),
                 interpolation=None,
                 aspect='auto',
                 extent=[phys_time[0], phys_time[-1],
                         phys_freq[0], phys_freq[-1]])
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="2%", pad=0.1)
    cb = fig.colorbar(im, cax=cax, orientation="vertical")
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=1, width=1, labelsize=7, pad=0.1)

    plt.savefig(opts.basename +
                f".data_convolved_sigma{sigv}.pdf",
                bbox_inches='tight')
    plt.close(fig)

    # get weight scaling from residual
    resc = data[mask] - data_convolved[mask]
    chival = np.vdot(resc, resc)

    # wgtc = wgt[mask] * opts.weight_scale
    datac = data[mask] - meandat
    wgtc = np.ones_like(datac)
    R = Mask(mask)

    sigmaf = np.std(datac)/2

    print(f"Using std of data to set sigmaf to {sigmaf}", file=log)


    # scale to lie in [0, 1]
    nu = phys_freq - phys_freq[0]
    nu /= nu.max()
    # nu = np.linspace(0, 1, nv)
    # t = np.linspace(0, 1, nt)
    t = phys_time - phys_time[0]
    t /= t.max()

    if opts.lt:
        tt = abs_diff(t, t)
        Kt = sigmaf**2 * np.exp(-tt**2/(2*opts.lt**2))
        Lt = np.linalg.cholesky(Kt + 1e-10*np.eye(nt))
    else:
        Lt = opts.sigmaf * np.eye(nt)

    vv = abs_diff(nu, nu)
    Kv = np.exp(-vv**2/(2*opts.lnu**2))
    Lv = np.linalg.cholesky(Kv + 1e-10*np.eye(nv))
    L = (Lv, Lt)
    LH = (Lv.T, Lt.T)

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(xsize, ysize))
    sols = np.zeros_like(data)

    def hess(x):
        return kv(LH, R.hdot(wgtc * R.dot(kv(L, x)))) + x

    delta = pcg(hess, kv(LH, R.hdot(wgtc * datac)),
                np.zeros((nv, nt), dtype=np.float64),
                minit=10, maxit=100, verbosity=2,
                report_freq=1, tol=1e-3)
    sol = kv(L, delta) + meandat

    data = R.hdot(datac) + meandat
    # import pdb; pdb.set_trace()
    res = np.where(mask, data-sol, 0.0)

    vmin = sol.min()
    vmax = sol.max()
    im = ax[0].imshow(data, vmin=vmin, vmax=vmax,
                            cmap='inferno', interpolation=None,
                            aspect='auto',
                            extent=[phys_time[0], phys_time[-1],
                                    phys_freq[0], phys_freq[-1]])
    ax[0].tick_params(axis='both', which='major',
                            length=1, width=1, labelsize=7)
    ax[0].set_xlabel('time / [hrs]')
    ax[0].set_ylabel('freq / [MHz]')

    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="2%", pad=0.1)
    cb = fig.colorbar(im, cax=cax, orientation="vertical")
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=1, width=1, labelsize=7, pad=0.1)

    im = ax[1].imshow(sol, vmin=vmin, vmax=vmax,
                            cmap='inferno', interpolation=None,
                            aspect='auto',
                            extent=[phys_time[0], phys_time[-1],
                                    phys_freq[0], phys_freq[-1]])
    ax[1].tick_params(axis='both', which='major',
                            length=1, width=1, labelsize=7)

    ax[1].set_ylabel('freq / [MHz]')
    divider = make_axes_locatable(ax[1])
    cax = divider.append_axes("right", size="2%", pad=0.1)
    cb = fig.colorbar(im, cax=cax, orientation="vertical")
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=1, width=1, labelsize=7, pad=0.1)

    im = ax[2].imshow(res, cmap='inferno',
                 vmin=0.25*res.min(), vmax=0.25*res.max(),
                 interpolation=None,
                 aspect='auto',
                 extent=[phys_time[0], phys_time[-1],
                         phys_freq[0], phys_freq[-1]])
    ax[2].tick_params(axis='both', which='major',
                            length=1, width=1, labelsize=7)
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="2%", pad=0.1)
    cb = fig.colorbar(im, cax=cax, orientation="vertical")
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=1, width=1, labelsize=7, pad=0.1)

    plt.savefig(opts.basename +
                f".th{opts.mad_threshold}_lnu{opts.lnu}_lt{opts.lt}.pdf",
                bbox_inches='tight')
    plt.close(fig)

    resc = res[mask]

    plt.hist(resc, bins=25)
    plt.savefig(opts.basename +
                f".th{opts.mad_threshold}_lnu{opts.lnu}_lt{opts.lt}_hist_resid.pdf",
                bbox_inches='tight')


    # fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True)
    # for c in range(4):
    #     # make lightcurves
    #     lc_raw = np.sum(data[c], axis=0)
    #     lw_raw = np.sum(wgt[c], axis=0)
    #     # lc_raw = np.where(lw_raw > 0, lc_raw/lw_raw, np.nan)

    #     # lstd = 1.0/np.sqrt(lw_raw[lw_raw!=0])

    #     lc_mean = np.sum(sols[c], axis=0)


    #     ax[c].plot(phys_time, lc_mean, 'k', alpha=0.75, linewidth=1)
    #     ax[c].plot(phys_time[lw_raw!=0], lc_raw[lw_raw!=0], '.r', alpha=0.15, markersize=3)
    #     ax[c].set_ylabel(f'{corrs[c]}')
    #     if c in [0,1,2]:
    #         ax[c].get_xaxis().set_visible(False)
    #     else:
    #         ax[c].set_xlabel('time / [hrs]')

    # plt.savefig(opts.basename +
    #             f".th{opts.mad_threshold}_lnu{opts.lnu}_lt{opts.lt}_lc.pdf",
    #             bbox_inches='tight')
    # plt.close(fig)

    # fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True)
    # for c in range(4):
    #     # make lightcurves
    #     lc_raw = np.sum(data[c]**2, axis=0)
    #     lw_raw = np.sum(wgt[c], axis=0)
    #     # lc_raw = np.where(lw_raw > 0, lc_raw/lw_raw, np.nan)

    #     # lstd = 1.0/np.sqrt(lw_raw[lw_raw!=0])

    #     lc_mean = np.sum(sols[c]**2, axis=0)


    #     ax[c].plot(phys_time, lc_mean, 'k', alpha=0.75, linewidth=1)
    #     ax[c].plot(phys_time[lw_raw!=0], lc_raw[lw_raw!=0], '.r', alpha=0.15, markersize=3)
    #     ax[c].set_ylabel(f'{corrs[c]}')
    #     if c in [0,1,2]:
    #         ax[c].get_xaxis().set_visible(False)
    #     else:
    #         ax[c].set_xlabel('time / [hrs]')

    # plt.savefig(opts.basename +
    #             f".th{opts.mad_threshold}_lnu{opts.lnu}_lt{opts.lt}_pw.pdf",
    #             bbox_inches='tight')
    # plt.close(fig)

    np.savez(opts.basename + '.npz', data=data, wgt=wgt, sols=sol, allow_pickle=True)
