import click
from jove.smoovie import cli
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('jove')
log = pyscilog.get_logger('ANIMATE')



@cli.command()
@click.option('-i', '--image',
              help="List of paths to restored images.fits")
@click.option('-o', '--outfile',
              help="Base name of output file")
def animate(**kw):
    args = OmegaConf.create(kw)
    from glob import glob
    args.image = glob(args.image)
    OmegaConf.set_struct(args, True)
    pyscilog.log_to_file(args.outfile + '.log')

    print("Input options :")
    for key in kw.keys():
        print('     %25s = %s' % (key, args[key]), file=log)

    import numpy as np
    from astropy.io import fits
    from astropy.wcs import WCS
    from datetime import datetime

    filelist = []
    timestamps = {}
    for i in range(len(args.image)):
        path = f"output/mfs/fits/os1{i:04d}.fits"
        filelist.append(path)
        hdr = fits.getheader(path)
        timestamps[path] = hdr['DATE-OBS']

    from multiprocessing.pool import Pool
    with Pool(processes=args.nthreads) as pool:
        dask.config.set(pool=pool)

