
import click
from jove.main import cli
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('jove')
log = pyscilog.get_logger('CUBE2FITS')

@cli.command()
@click.option("-d", "--data", type=str, required=True,
              help="Path to data.zarr")
@click.option("-o", "--outfile", type=str, required=True,
              help='Base name of output file.')
@click.option('-nthreads', '--nthreads', type=int, default=64,
              help='Number of dask threads.')
@click.option('-cell', '--cell-size', type=float, default=1.0,
              help='Cell size in degrees.')
def cube2fits(**kw):
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
    import dask.array as da
    import dask
    from dask.diagnostics import ProgressBar
    from jove.utils import cube2fits

    from multiprocessing.pool import Pool
    with Pool(processes=args.nthreads) as pool:
        dask.config.set(pool=pool)

        Din = xr.open_dataset(args.data, chunks={'time':1, 'nx':-1, 'ny':-1}, engine='zarr')

        image = Din.image.data
        ras = Din.ras.data
        decs = Din.decs.data
        times = Din.times.data
        idx = da.arange(times.size, chunks=times.chunks)

        t = da.blockwise(
                cube2fits, 't',
                args.outfile, None,
                image, 'txy',
                ras, 't',
                decs, 't',
                times, 't',
                np.ones(1), None,
                args.cell_size, None,
                idx, 't',
                dtype=times.dtype
        )

        with ProgressBar():
            dask.compute(t)