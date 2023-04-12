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



    # load data
    basename = opts.basename.rstrip('.norm.fits')
    std = fits.getdata(basename + '.std.fits')[0]
    data = fits.getdata(opts.basename)[0].astype(np.float64)
    wgt = np.where(data != 0, 1.0/std**2, 0.0)
    nv, nt = data.shape
    hdr = fits.getheader(opts.basename)
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
    nu /= nu.max()
    # nu = np.linspace(0, 1, nv)
    # t = np.linspace(0, 1, nt)
    t = phys_time - phys_time[0]
    t /= t.max()

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
    print(f"mean of data = {meandat}", file=log)

    # construct convolution kernel (LB - may need padding here if the kernel is large)
    def kernel(x, l):
        nx = x.size
        return np.exp(-(x-x[nx//2])**2/(2*l**2))

    kt = kernel(t, opts.lt)
    kv = kernel(nu, opts.lnu)
    K = kv[:, None] * kt[None, :]

    K /= K.sum()
    Khat = r2c(iFs(K), axes=(0,1), nthreads=opts.nthreads, forward=True, inorm=0)

    data = np.where(mask, data, 0.0)

    dhat = r2c(iFs(data), axes=(0,1), nthreads=opts.nthreads, forward=True, inorm=0)
    dhat *= Khat
    data_convolved = Fs(c2r(dhat, axes=(0,1), nthreads=opts.nthreads, forward=False, inorm=2, lastsize=nt))

    res = np.where(mask, data - data_convolved, 0.0)

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(xsize, ysize))
    im = ax[0].imshow(data, cmap='inferno',
                 vmin=data_convolved.min(), vmax=data_convolved.max(),
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

    im = ax[1].imshow(data_convolved, cmap='inferno',
                 vmin=data_convolved.min(), vmax=data_convolved.max(),
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

    im = ax[2].imshow(res, cmap='inferno',
                 vmin=0.25*res.min(), vmax=0.25*res.max(),
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

    plt.savefig(basename +
                f"..th{opts.mad_threshold}_lnu{opts.lnu}_lt{opts.lt}_convolved.pdf",
                bbox_inches='tight')
    plt.close(fig)

    # get weight scaling from residual
    resc = data[mask] - data_convolved[mask]

    plt.hist(resc, bins=25)
    plt.title('Hist convolved resid')
    plt.savefig(basename +
                f".th{opts.mad_threshold}_lnu{opts.lnu}_lt{opts.lt}_convolved_hist_resid.pdf",
                bbox_inches='tight')
    plt.close()

    # wgtc = wgt[mask] * opts.weight_scale
    datac = data[mask]
    wgtc = np.ones_like(datac)
    R = Mask(mask)

    sigmaf = np.std(datac)/2

    print(f"Using std of data to set sigmaf to {sigmaf}", file=log)

    if opts.lt:
        tt = abs_diff(t, t)
        Kt = sigmaf**2 * np.exp(-tt**2/(2*opts.lt**2))
        Lt = np.linalg.cholesky(Kt + 1e-10*np.eye(nt))
    else:
        Lt = sigmaf * np.eye(nt)

    vv = abs_diff(nu, nu)
    Kv = np.exp(-vv**2/(2*opts.lnu**2))
    Lv = np.linalg.cholesky(Kv + 1e-10*np.eye(nv))
    L = (Lv, Lt)
    LH = (Lv.T, Lt.T)

    fig, ax = plt.subplots(nrows=3, ncols=1, figsize=(xsize, ysize))
    sols = np.zeros_like(data)

    def hess(x):
        return kron_matvec(LH, R.hdot(wgtc * R.dot(kron_matvec(L, x)))) + x

    delta = pcg(hess, kron_matvec(LH, R.hdot(wgtc * datac)),
                np.zeros((nv, nt), dtype=np.float64),
                minit=10, maxit=100, verbosity=2,
                report_freq=1, tol=1e-3)
    sol = kron_matvec(L, delta)

    data = R.hdot(datac)
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
    ax[2].set_xlabel('time / [hrs]')
    ax[2].set_ylabel('freq / [MHz]')
    divider = make_axes_locatable(ax[2])
    cax = divider.append_axes("right", size="2%", pad=0.1)
    cb = fig.colorbar(im, cax=cax, orientation="vertical")
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=1, width=1, labelsize=7, pad=0.1)

    fig.suptitle('Smoothed by GPR')

    plt.savefig(basename +
                f".th{opts.mad_threshold}_lnu{opts.lnu}_lt{opts.lt}.pdf",
                bbox_inches='tight')
    plt.close(fig)

    resc = res[mask]

    plt.hist(resc, bins=25)
    plt.title('Hist resid')
    plt.savefig(basename +
                f".th{opts.mad_threshold}_lnu{opts.lnu}_lt{opts.lt}_hist_resid.pdf",
                bbox_inches='tight')
    plt.close()


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

    # plt.savefig(basename +
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

    # plt.savefig(basename +
    #             f".th{opts.mad_threshold}_lnu{opts.lnu}_lt{opts.lt}_pw.pdf",
    #             bbox_inches='tight')
    # plt.close(fig)

    np.savez(basename + f'.th{opts.mad_threshold}_lnu{opts.lnu}_lt{opts.lt}' + '.npz',
             data=data, wgt=wgt, sols=sol, residual=res,
             time=phys_time, freq=phys_freq, allow_pickle=True)
