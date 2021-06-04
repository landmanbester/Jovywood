
import click
from jove.main import cli
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('jove')
log = pyscilog.get_logger('FIT')

@cli.command()
@click.option("-d", "--data", type=str,
              help="Path to data.zarr")
@click.option("-o", "--outfile", type=str, default='pfb',
              help='Base name of output file.')
@click.option("-pc", "--pix_chunks", type=int, default=50,
              help='Pixel chunks')
@click.option('-nthreads', '--nthreads', type=int, default=64,
              help='Number of dask threads.')
def fit(**kw):
    '''
    Fit hyperparameters for each pixel and write it out as a zarr array
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
    from jove.utils import fit_pix

    from multiprocessing.pool import Pool
    with Pool(processes=args.nthreads) as pool:
        dask.config.set(pool=pool)

        Din = xr.open_dataset(args.data, chunks={'time':-1, 'nx':args.pix_chunks, 'ny':args.pix_chunks}, engine='zarr')

        image = Din.image.data
        rmss = Din.rmss.data
        times = Din.times.data

        nt, nx, ny = image.shape

        # normalise to between 0 and 1
        t = times - times.min()
        t = t/t.max()

        # precompute abs diffs
        xxsq = abs_diff(t, t)

        Sigma = rmss**2
        sigman0 = 1.0

        thetas = da.blockwise(
            fit_pix, 'pxy',
            image, 'txy',
            xxsq, None,
            Sigma, None,
            sigman0, None,
            new_axes={"p": 3},
            align_arrays=False,
            dtype=image.dtype
        )

        data_vars = {'theta':(('three', 'nx', 'ny'), thetas)}

        Dout = xr.Dataset(data_vars)

        with ProgressBar():
            Dout.to_zarr(args.outfile + '_hypers.zarr', mode='w', compute=True)