import click
from jove.dspec import cli
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('jove')
log = pyscilog.get_logger('MSTD')

@cli.command()
@click.option("-bn", "--basename", type=str, required=True,
              help="Path to folder containing OFF/data.npz")
@click.option("-o", "--outfile", type=str, required=True,
              help='Base name of output file.')
@click.option('-nthreads', '--nthreads', type=int, default=64,
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
    os.environ["OMP_NUM_THREADS"] = str(args.nthreads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.nthreads)
    os.environ["MKL_NUM_THREADS"] = str(args.nthreads)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(args.nthreads)
    os.environ["NUMBA_NUM_THREADS"] = str(args.nthreads)
    import numpy as np
    import matplotlib as mpl
    mpl.rcParams.update({'font.size': 12, 'font.family': 'serif'})
    import matplotlib.pyplot as plt
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    from glob import glob

    search_name = args.basename.rstrip('/') + '/OFF/*.npz'
    sources_list = glob(search_name)
    nsource = len(sources_list)
    if not nsource:
        raise ValueError(f"No sources found at {search_name}")
    else:
        print(f"Found {nsource} sources", file=log)

    # first the mean
    means = []
    for s in sources_list:
        data = np.load(s)
        means.append(data['sols'])[None]

    means = np.concatenate(means)

    mean = np.mean(means, axis=0)
    std = np.std(means, axis=0)



    nv, nt = mean[0].shape

    ysize = 12
    xsize = int(np.ceil(4 * nt * ysize/nv))

    fig, ax = plt.subplots(nrows=1, ncols=4, figsize=(xsize, ysize))
    corrs = ['I', 'Q', 'U', 'V']
    for c in range(4):
        meanc = mean[c]
        stdc = std[c]

        ax[c].set_title(f"{corrs[c]}")
        vmin = stdc.min()
        vmax = stdc.max()
        im = ax[c].imshow(stdc, vmin=vmin, vmax=vmax,
                             cmap='inferno', interpolation=None,
                             aspect='auto')
        ax[c].tick_params(axis='both', which='major',
					      length=1, width=1, labelsize=7)

        divider = make_axes_locatable(ax[c])
        cax = divider.append_axes("bottom", size="10%", pad=0.05)
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=1, width=1, labelsize=7, pad=0.05)

    plt.savefig(args.basename + f".th{args.mad_threshold}_lnu{args.lnu}_lt{args.lt}.pdf",
                bbox_inches='tight')
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
        ax[c].set_ylabel(f'{corrs[c]}')
        if c in [0,1,2]:
            ax[c].get_xaxis().set_visible(False)
        else:
            ax[c].set_xlabel('time / [hrs]')

    plt.savefig(args.basename + args.source +
                f".th{args.mad_threshold}_lnu{args.lnu}_lt{args.lt}_lc.pdf",
                bbox_inches='tight')
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
        ax[c].set_ylabel(f'{corrs[c]}')
        if c in [0,1,2]:
            ax[c].get_xaxis().set_visible(False)
        else:
            ax[c].set_xlabel('time / [hrs]')

    plt.savefig(args.basename + args.source +
                f".th{args.mad_threshold}_lnu{args.lnu}_lt{args.lt}_pw.pdf",
                bbox_inches='tight')
    plt.close(fig)

    np.savez(args.basename + args.source + '.npz', data=data, wgt=wgt, sols=sols, allow_pickle=True)