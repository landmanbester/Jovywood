import click
from jove.dspec import cli
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('jove')
log = pyscilog.get_logger('GPRS')

from scabha.schema_utils import clickify_parameters
from jove.parser.schemas import schema

# create default parameters from schema
defaults = {}
for key in schema.gpr_smooth["inputs"].keys():
    defaults[key] = schema.gpr_smooth["inputs"][key]["default"]

@cli.command(context_settings={'show_default': True})
@clickify_parameters(schema.gpr_smooth)
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
    defaults.update(kw)
    opts = OmegaConf.create(defaults)
    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    pyscilog.log_to_file(f'gpr_smooth_{timestamp}.log')

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
    import xarray as xr
    from astropy.io import fits
    from jove.utils import madmask, Mask
    from africanus.gps.utils import abs_diff
    from pfb.utils.misc import kron_matvec
    from pfb.opt.pcg import pcg
    from pfb.utils.fits import data_from_header
    from ducc0.fft import r2c, c2r
    iFs = np.fft.ifftshift
    Fs = np.fft.fftshift
    import matplotlib as mpl
    mpl.rcParams.update({'font.size': 18, 'font.family': 'serif'})
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import scipy
    from pathlib import Path


    # load data
    stokes = opts.basename[-1].upper()
    idx = 'IQUV'.find(stokes)
    if str(idx) not in '0123':
        print(f'{stokes} is not a valid Stokes parameter', file=log)
        quit()
    try:
        data = fits.getdata(opts.basename + '.fits')[idx].astype(np.float64)
    except Exception as e:
        print(f"Failed to load data at {opts.basename + '.fits'}")
        quit()
    data = np.flipud(data)
    norm = fits.getdata(opts.basename + '.norm.fits')[0].astype(np.float64)
    norm = np.flipud(norm)
    std = fits.getdata(opts.basename + '.std.fits')[0].astype(np.float64)
    std = np.flipud(std)
    # norm should be data/std
    mask = data != 0
    assert np.allclose(norm[mask], data[mask]/std[mask])
    wgt = np.where(data != 0, 1.0/std, 0.0)
    nv, nt = data.shape
    hdr = fits.getheader(opts.basename + '.fits')
    delt = hdr['CDELT1']
    phys_time, ref_time = data_from_header(hdr, axis=1)
    phys_time -= phys_time.min()
    phys_time /= 3600  # sec to hr
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

    # scale to lie in [0, 1]
    nu = phys_freq - phys_freq[0]
    lnu = opts.lnu/nu.max()
    nu /= nu.max()
    t = phys_time - phys_time[0]
    lt = opts.lt / t.max() / 3600
    t /= t.max()

    print(f'Smoothing kernel widths: time {opts.lt} s, freq {opts.lnu} MHz', file=log)
    # get directory where fits files live
    outpath = Path(opts.basename).resolve().parent
    outpath = Path(str(outpath) + f'/{stokes}/lnu{opts.lnu}MHz_lt{opts.lt}s')
    outpath.mkdir(parents=True, exist_ok=True)

    # refine mask
    sigv = 3
    sigt = 3
    mask = madmask(data, wgt, th=opts.mad_threshold, sigv=sigv, sigt=sigt).astype(np.bool)

    ysize = 16
    xsize = int(np.ceil(nt * 12/nv))
    meandat = np.mean(data[mask])
    print(f"mean of data = {meandat}", file=log)

    # construct convolution kernel (LB - may need padding here if the kernel is large)
    def kernel(x, l):
        nx = x.size
        return np.exp(-(x-x[nx//2])**2/(2*l**2))

    print('Convolving', file=log)
    kt = kernel(t, lt)
    kv = kernel(nu, lnu)
    K = kv[:, None] * kt[None, :]

    K /= K.sum()
    Khat = r2c(iFs(K), axes=(0,1), nthreads=opts.nthreads, forward=True, inorm=0)

    norm = np.where(mask, norm, 0.0)

    dhat = r2c(iFs(norm), axes=(0,1), nthreads=opts.nthreads, forward=True, inorm=0)
    dhat *= Khat
    data_convolved = Fs(c2r(dhat, axes=(0,1), nthreads=opts.nthreads, forward=False, inorm=2, lastsize=nt))
    mhat = r2c(iFs(mask.astype(np.float64)), axes=(0,1), nthreads=opts.nthreads, forward=True, inorm=0)
    mhat *= Khat
    mask_convolved = Fs(c2r(mhat, axes=(0,1), nthreads=opts.nthreads, forward=False, inorm=2, lastsize=nt))

    scaled_result = np.zeros_like(data_convolved)
    scaled_result[mask] = data_convolved[mask]/mask_convolved[mask]

    res_conv = np.where(mask, norm - scaled_result, 0.0)
    norm_mad = scipy.stats.median_abs_deviation(norm[mask], scale='normal')
    norm_med = np.median(norm[mask])
    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(xsize, ysize))
    im = ax[0].imshow(norm, cmap='inferno',
                 vmin=norm_med - norm_mad, vmax=norm_med + norm_mad,
                 interpolation=None,
                 aspect='auto',
                 extent=[phys_time[0], phys_time[-1],
                         phys_freq[0], phys_freq[-1]])
    ax[0].tick_params(axis='both', which='major',
                            length=1, width=1, labelsize=7)
    ax[0].set_ylabel('freq / [MHz]')
    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="2%", pad=0.1)
    cb = fig.colorbar(im, cax=cax, orientation="vertical")
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=1, width=1, labelsize=7, pad=0.1)

    conv_mad = scipy.stats.median_abs_deviation(scaled_result[mask], scale='normal')
    conv_med = np.median(scaled_result[mask])
    im = ax[1].imshow(scaled_result,
                      cmap='inferno',
                      vmin=conv_med - 2*conv_mad, vmax=conv_med + 2*conv_mad,
                      interpolation=None,
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

    res_mad = scipy.stats.median_abs_deviation(res_conv[mask], scale='normal')
    res_med = np.median(res_conv[mask])
    im = ax[2].imshow(res_conv, cmap='inferno',
                 vmin=res_med - res_mad, vmax=res_med + res_mad,
                 interpolation=None,
                 aspect='auto',
                 extent=[phys_time[0], phys_time[-1],
                         phys_freq[0], phys_freq[-1]])
    ax[2].tick_params(axis='both', which='major',
                            length=1, width=1, labelsize=7)
    ax[2].set_xlabel('time / [hrs]')
    ax[2].set_ylabel('freq / [MHz]')
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="2%", pad=0.1)
    cb = fig.colorbar(im, cax=cax, orientation="vertical")
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=1, width=1, labelsize=7, pad=0.1)

    fig.suptitle('Smoothed by convolvolution')

    plt.savefig(str(outpath) + "/convolved.pdf",
                bbox_inches='tight')
    plt.close(fig)

    plt.hist(res_conv[mask], bins=25)
    plt.title('Hist convolved resid')
    plt.savefig(str(outpath) + "/convolved_hist_resid.pdf",
                bbox_inches='tight')
    plt.close()

    datac = data[mask]
    wgtc = wgt[mask]
    R = Mask(mask)

    sigmaf = np.std(datac)*5

    print(f"Using std of data to set sigmaf to {sigmaf}", file=log)

    if opts.lt:
        tt = abs_diff(t, t)
        Kt = sigmaf**2 * np.exp(-tt**2/(2*lt**2))
        Lt = np.linalg.cholesky(Kt + 1e-10*np.eye(nt))
    else:
        Lt = sigmaf * np.eye(nt)

    vv = abs_diff(nu, nu)
    Kv = np.exp(-vv**2/(2*lnu**2))
    Lv = np.linalg.cholesky(Kv + 1e-10*np.eye(nv))
    L = (Lv, Lt)
    LH = (Lv.T, Lt.T)

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(xsize, ysize))
    sols = np.zeros_like(data)

    def hess(x):
        return kron_matvec(LH, R.hdot(wgtc * R.dot(kron_matvec(L, x)))) + x

    print('Running GPR', file=log)
    delta = pcg(hess, kron_matvec(LH, R.hdot(wgtc * datac)),
                np.zeros((nv, nt), dtype=np.float64),
                minit=10, maxit=100, verbosity=2,
                report_freq=1, tol=1e-2)
    sol = kron_matvec(L, delta)

    data = R.hdot(datac)
    # import pdb; pdb.set_trace()
    res_sol = np.where(mask, data-sol, 0.0)

    data_mad = scipy.stats.median_abs_deviation(data[mask], scale='normal')
    data_med = np.median(data[mask])
    im = ax[0].imshow(data,
                      vmin=data_med - data_mad, vmax=data_med + data_mad,
                            cmap='inferno', interpolation=None,
                            aspect='auto',
                            extent=[phys_time[0], phys_time[-1],
                                    phys_freq[0], phys_freq[-1]])
    ax[0].tick_params(axis='both', which='major',
                            length=1, width=1, labelsize=7)
    ax[0].set_ylabel('freq / [MHz]')

    divider = make_axes_locatable(ax[0])
    cax = divider.append_axes("right", size="2%", pad=0.1)
    cb = fig.colorbar(im, cax=cax, orientation="vertical")
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=1, width=1, labelsize=7, pad=0.1)
    sol_mad = scipy.stats.median_abs_deviation(sol[mask], scale='normal')
    sol_med = np.median(sol[mask])
    im = ax[1].imshow(sol,
                      vmin=sol_med - 2*sol_mad, vmax=sol_med + 2*sol_mad,
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
    res_mad = scipy.stats.median_abs_deviation(res_sol[mask], scale='normal')
    res_med = np.median(res_sol[mask])
    im = ax[2].imshow(res_sol, cmap='inferno',
                 vmin=res_med - res_mad, vmax=res_med + res_mad,
                 interpolation=None,
                 aspect='auto',
                 extent=[phys_time[0], phys_time[-1],
                         phys_freq[0], phys_freq[-1]])
    ax[2].tick_params(axis='both', which='major',
                            length=1, width=1, labelsize=7)
    ax[2].set_xlabel('time / [hrs]')
    ax[2].set_ylabel('freq / [MHz]')
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="2%", pad=0.1)
    cb = fig.colorbar(im, cax=cax, orientation="vertical")
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=1, width=1, labelsize=7, pad=0.1)

    fig.suptitle('Smoothed by GPR')

    plt.savefig(str(outpath) + "/gpr.pdf",
                bbox_inches='tight')
    plt.close(fig)

    resc = res_sol[mask]

    plt.hist(resc, bins=25)
    plt.title('Hist resid')
    plt.savefig(str(outpath) + "/gpr_hist_resid.pdf",
                bbox_inches='tight')
    plt.close()


    hdr = fits.getheader(opts.basename + '.fits')
    fits.writeto(str(outpath) + "/gpr.fits",
                 np.tile(np.flipud(sol), (4,1,1)), overwrite=True)

    fits.writeto(str(outpath) + "/convolved.fits",
                 np.tile(np.flipud(scaled_result), (4,1,1)), overwrite=True)

    np.savez(str(outpath) + '/results.npz',
             data=data, wgt=wgt, sols=sol, residual_gpr=res_sol,
             norm=norm, convolved=scaled_result, residual_convolved=res_conv,
             time=phys_time, freq=phys_freq, lnu=opts.lnu, lt=opts.lt, allow_pickle=True)
