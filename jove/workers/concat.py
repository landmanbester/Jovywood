
import click
from jove.main import cli
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('jove')
log = pyscilog.get_logger('CONCAT')

@cli.command()
@click.option('-i', '--image',
              help="List of paths to restored images.fits")
@click.option('-r', '--resid',
              help="List of paths to residual images.fits")
@click.option('-o', '--output',
              help="Base name of output file")
@click.option('-pc', '--pix-chunks',
              help="Pixel chunks")
def concat(**kw):
    '''
    Concatenate a bunch of fits files and write out a chunked
    zarr cube.
    '''
    args = OmegaConf.create(kw)
    OmegaConf.set_struct(args, True)
    pyscilog.log_to_file(args.output + '.log')
    pyscilog.enable_memory_logging(level=3)

    print("Input options :")
    for key in kw.keys():
        print('     %25s = %s' % (key, args[key]), file=log)

    import numpy as np
    from astropy.io import fits
    from datetime import datetime
    from jove.utils import load_fits

    nimages = len(args.image)
    assert len(args.image) == len(args.resid)

    image_cube = []
    times = np.zeros(nimages)
    rmss = np.zeros(nimages)
    ras = np.zeros(nimages)
    decs = np.zeros(nimages)
    nx = None
    ny = None
    nband = None
    ncorr = None
    for k, image in enumerate(args.image):
        hdr = fits.getheader(image)
        rhdr = fits.getheader(args.resid[k])
        if nx is None:
            nx = hdr['NAXIS1']

        assert nx == hdr['NAXIS1']
        assert nx == rhdr['NAXIS1']

        if ny is None:
            ny = hdr['NAXIS2']

        assert ny == hdr['NAXIS2']
        assert ny == rhdr['NAXIS2']

        if nband is None:
            nband = hdr['NAXIS3']

        assert nband == hdr['NAXIS3']
        assert nband == rhdr['NAXIS3']

        if ncorr is None:
            ncorr = hdr['NAXIS4']

        assert ncorr == hdr['NAXIS4']
        assert ncorr == rhdr['NAXIS4']

        times[k] = init_datetime(hdr['DATE-OBS'])
        assert times[k] == init_datetime(rhdr['DATE-OBS'])

        ras[k] = hdr['CRVAL1']
        assert ras[k] == rhdr['CRVAL1']

        decs[k] = hdr['CRVAL2']
        assert decs[k] == rhdr['CRVAL2']

        rmss[k] = np.std(load_fits(args.resid[k])[0])
        image_cube.append(load_fits(image)[0])

        print(k, times[k])

    image_cube = np.concatenate(image_cube, axis=0)
    import dask.array as da
    image_cube = da.from_array(image_cube,
                               chunks=(1, args.pix_chunks, args.pix_chunks),
                               name=False)
    times = da.from_array(times,
                          chunks=1,
                          name=False)
    rmss = da.from_array(rmss,
                         chunks=1,
                         name=False)

    ras = da.from_array(ras,
                        chunks=1,
                        name=False)

    decs = da.from_array(decs,
                         chunks=1,
                         name=False)

    import xarray as xr
    data_vars = {'image':(('time', 'nx', 'ny'), image_cube)}
    coords = {'times': (('time',), times),
              'rmss': (('time',), rmss),
              'ras': (('time',), ras),
              'decs': (('time',), decs)}

    D = xr.Dataset(data_vars, coords)
    D.to_zarr(args.outfile)