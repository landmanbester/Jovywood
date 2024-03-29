
import click
from jove.smoovie import cli
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('jove')
log = pyscilog.get_logger('INTERP')

@cli.command()
@click.option("-d", "--data", type=str, required=True,
              help="Path to data.zarr")
@click.option("-h", "--hypers", type=str, required=True,
              help="Path to hypers.zarr")
@click.option("-o", "--outfile", type=str, required=True,
              help='Base name of output file.')
@click.option("-pc", "--pix-chunks", type=int, default=1000,
              help='Pixel chunks')
@click.option("-poc", "--pix-out-chunks", type=int, default=100,
              help='Pixel chunks')
@click.option('-nto', "--ntime-out", type=int, required=True,
              help="Number of output times")
@click.option('-os', "--oversmooth", type=float, default=1,
              help="Over-smoothing factor.")
@click.option('-nthreads', '--nthreads', type=int, default=64,
              help='Number of dask threads.')
def interp(**kw):
    '''
    Interpolate the image using GPR
    '''
    args = OmegaConf.create(kw)
    OmegaConf.set_struct(args, True)
    pyscilog.log_to_file(args.outfile + '.log')
    pyscilog.enable_memory_logging(level=3)

    print("Input options :")
    for key in kw.keys():
        print('     %25s = %s' % (key, args[key]), file=log)

    import os
    os.environ["OMP_NUM_THREADS"] = str(1)
    os.environ["OPENBLAS_NUM_THREADS"] = str(1)
    os.environ["MKL_NUM_THREADS"] = str(1)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(1)
    os.environ["NUMBA_NUM_THREADS"] = str(1)
    import numpy as np
    import xarray as xr
    from jove.utils import abs_diff
    import dask.array as da
    import dask
    from dask.diagnostics import ProgressBar
    from PIL import Image
    from glob import glob
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from scipy.interpolate import interp1d
    from jove.utils import interp_pix

    from multiprocessing.pool import ThreadPool
    dask.config.set(pool=ThreadPool(processes=args.nthreads))

    Din = xr.open_dataset(args.data, chunks={'time':-1,
                                            'nx':args.pix_chunks,
                                            'ny':args.pix_chunks},
                                            engine='zarr')

    image = Din.image.data
    rmss = Din.rmss.data
    times = Din.times.data.compute()
    ras = Din.ras.data.compute()
    decs = Din.decs.data.compute()

    thetas = xr.open_dataset(args.hypers, chunks={'time':-1,
                                                    'nx':args.pix_chunks,
                                                    'ny':args.pix_chunks},
                                                    engine='zarr').theta.data

    ntime, nx, ny = image.shape

    # normalise to between 0 and 1
    tmin = times.min()
    tmax = times.max()
    t = (times - tmin)/(tmax - tmin)

    raso = interp1d(t, ras, kind='cubic', assume_sorted=True)
    decso = interp1d(t, decs, kind='cubic', assume_sorted=True)

    # precompute abs diffs
    xxsq = abs_diff(t, t)
    tp = np.linspace(0, 1, args.ntime_out)
    xxpsq = abs_diff(tp, t)

    Sigma = rmss**2

    image_out = da.blockwise(
        interp_pix, 'txy',
        thetas, 'pxy',
        image, 'txy',
        xxsq, None,
        xxpsq, None,
        Sigma, None,
        args.oversmooth, None,
        align_arrays=False,
        adjust_chunks={'t': args.ntime_out},
        dtype=image.dtype
    )

    tout = tmin + tp*(tmax - tmin)
    tout = da.from_array(tout, chunks=1)
    rasout = da.from_array(raso(tp), chunks=1)
    decsout = da.from_array(decso(tp), chunks=1)

    data_vars = {'image':(('time', 'nx', 'ny'),
                 image_out.rechunk((1, args.pix_out_chunks, args.pix_out_chunks)))}
    coords = {'times': (('time',), tout),
                'ras': (('time'), rasout),
                'decs': (('time'), decsout)}

    Dout = xr.Dataset(data_vars, coords)
    with ProgressBar():
        Dout.to_zarr(args.outfile + '_os' + str(args.oversmooth) + '.zarr', mode='w', compute=True)

    # print('2png')
    # imgs = []
    # for i in range(args.ntime_out):
    #     plt.figure()
    #     plt.imshow(imagep[i].T, cmap='inferno', vmin=1e-6, vmax=0.15, origin='lower')
    #     plt.title(str(i))
    #     plt.savefig(args.outfile + str(i) + '.png', dpi=300)
    #     imgs.append(args.outfile + str(i) + '.png')
    #     plt.close()

    # print('2gif')
    # frames = []
    # for i in imgs:
    #     new_frame = Image.open(i)
    #     frames.append(new_frame)

    # frames[0].save(args.outfile + '.gif', format='GIF',
    #             append_images=frames[1:],
    #             save_all=True,
    #             duration=args.duration*1000/args.ntime_out, loop=0)