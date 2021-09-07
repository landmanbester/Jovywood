import click
from jove.dspec import cli
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('jove')
log = pyscilog.get_logger('GPRS')

@cli.command()
@click.option("-bn", "--basename", type=str, required=True,
              help="Path to folder containing TARGET/data.fits")
@click.option("-s", "--source", type=str, required=True,
              help="source nae inside basename")
@click.option("-th", "--mad-threshold", type=float, default=6.0,
              help="Threshold for MAD flagging")
@click.option('-sigmaf', '--sigmaf', type=float, default=0.01,
              help="Zero mode of GP prior")
@click.option('-lnu', '--lnu', type=float, default=0.25,
              help="Length scale of freq covariance function")
@click.option('-lt', '--lt', type=float, default=0.0001,
              help="Length scale of time covariance function")
@click.option("-ws", "--weight-scale", type=float, default=10,
              help="Weight scaling factor")
@click.option("-o", "--outfile", type=str, required=True,
              help='Base name of output file.')
@click.option('-nthreads', '--nthreads', type=int, default=64,
              help='Number of dask threads.')
def gpr_smooth(**kw):
    '''
    smooth dynamic spectra with GP
    '''
    args = OmegaConf.create(kw)
    if not args.basename.endswith('/'):
        args.basename += '/'
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
    from jove.utils import madmask
    from pfb.operators.mask import Mask
    from africanus.gps.utils import abs_diff
    from pfb.utils.misc import kron_matvec as kv
    from pfb.opt.pcg import pcg
    import matplotlib as mpl
    mpl.rcParams.update({'font.size': 8, 'font.family': 'serif'})
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable



    # Preparing the filename string for store results
    # basename = "/home/landman/Data/MeerKAT/Jove/dspec/FullJupiter_NoSubTarget_SourceOff/"
    # source = 'TARGET/1608538564_20:09:36.999_-20:26:47.350.fits'
    wgt = fits.getdata(args.basename + 'Weights.fits')  # select Stokes I as example
    data = fits.getdata(args.basename + args.source)
    nv, nt = data[0].shape

    hdr = fits.getheader(args.basename + args.source)
    delt = hdr['CDELT1']  # delta t in sec
    phys_time = np.arange(nt) * delt/3600  # sec to hr

    # refine mask
    sigv = 3
    sigt = 3
    mask = madmask(data, wgt, th=args.mad_threshold, sigv=sigv, sigt=sigt)
    wgt[:, mask] = 0.0
    data[:, mask] = 0.0
    wgt *= args.weight_scale
    R = Mask(mask)

    nu = np.linspace(0, 1, nv)
    t = np.linspace(0, 1, nt)

    if args.lt:
        tt = abs_diff(t, t)
        Kt = args.sigmaf**2 * np.exp(-tt**2/(2*args.lt**2))
        Lt = np.linalg.cholesky(Kt + 1e-10*np.eye(nt))
    else:
        Lt = args.sigmaf * np.eye(nt)

    vv = abs_diff(nu, nu)
    Kv = np.exp(-vv**2/(2*args.lnu**2))
    Lv = np.linalg.cholesky(Kv + 1e-10*np.eye(nv))
    L = (Lv, Lt)
    LH = (Lv.T, Lt.T)

    fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(24, 6))
    corrs = ['I', 'Q', 'U', 'V']
    sols = np.zeros_like(data)
    for c in range(4):
        wgtc = wgt[c, ~mask]
        datac = data[c, ~mask]

        def hess(x):
            return kv(LH, R.hdot(wgtc * R.dot(kv(L, x)))) + x

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
        ax[0, c].axis('off')
        divider = make_axes_locatable(ax[0, c])
        cax = divider.append_axes("bottom", size="10%", pad=0.1)
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=1, width=1, labelsize=5, pad=0.1)



        im = ax[1, c].imshow(sol, vmin=-0.005, vmax=0.005,
                             cmap='inferno', interpolation=None)
        ax[1, c].axis('off')
        divider = make_axes_locatable(ax[1, c])
        cax = divider.append_axes("bottom", size="10%", pad=0.05)
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=1, width=1, labelsize=5, pad=0.05)

        sols[c] = sol

    plt.savefig(args.basename + args.source +
                f".th{args.mad_threshold}_lnu{args.lnu}_lt{args.lt}.png",
                dpi=200, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True)
    for c in range(4):
        # make lightcurves
        lc_raw = np.sum(data[c], axis=0)
        lw_raw = np.sum(wgt[c], axis=0)
        # lc_raw = np.where(lw_raw > 0, lc_raw/lw_raw, np.nan)

        # lstd = 1.0/np.sqrt(lw_raw[lw_raw!=0])

        lc_mean = np.sum(sols[c], axis=0)


        ax[c].plot(phys_time, lc_mean, 'k', alpha=0.75, linewidth=1)
        ax[c].plot(phys_time[lw_raw!=0], lc_raw[lw_raw!=0], '.r', alpha=0.15, markersize=3)
        if c in [0,1,2]:
            ax[c].get_xaxis().set_visible(False)
        else:
            ax[c].set_xlabel('time / [hrs]')

    plt.savefig(args.basename + args.source +
                f".th{args.mad_threshold}_lnu{args.lnu}_lt{args.lt}_lc.png",
                dpi=200, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(nrows=4, ncols=1, sharex=True)
    for c in range(4):
        # make lightcurves
        lc_raw = np.sum(data[c]**2, axis=0)
        lw_raw = np.sum(wgt[c], axis=0)
        # lc_raw = np.where(lw_raw > 0, lc_raw/lw_raw, np.nan)

        # lstd = 1.0/np.sqrt(lw_raw[lw_raw!=0])

        lc_mean = np.sum(sols[c]**2, axis=0)


        ax[c].plot(phys_time, lc_mean, 'k', alpha=0.75, linewidth=1)
        ax[c].plot(phys_time[lw_raw!=0], lc_raw[lw_raw!=0], '.r', alpha=0.15, markersize=3)
        if c in [0,1,2]:
            ax[c].get_xaxis().set_visible(False)
        else:
            ax[c].set_xlabel('time / [hrs]')

    plt.savefig(args.basename + args.source +
                f".th{args.mad_threshold}_lnu{args.lnu}_lt{args.lt}_pw.png",
                dpi=200, bbox_inches='tight')
    plt.close(fig)