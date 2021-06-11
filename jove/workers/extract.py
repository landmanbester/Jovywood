import click
from jove.main import cli
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('jove')
log = pyscilog.get_logger('EXTRACT')

@cli.command()
@click.option("-d", "--data", type=str, required=True,
                   help="Path to data.zarr")
@click.option('-t0', '--t0', type=int, default=0)
@click.option('-tf', '--tf', type=int, default=-1)
@click.option('-x0', '--x0', type=int)
@click.option('-xf', '--xf', type=int)
@click.option('-y0', '--y0', type=int)
@click.option('-yf', '--yf', type=int)
@click.option('-o', '--output', required=True,
              help="Base name of output file")
@click.option('-pc', '--pix-chunks',
              help="Pixel chunks")
@click.option('-dur', "--duration", type=float, default=5,
              help="Duration of gif in s")
def extract(**kw):
    '''
    Extract a region of the zarrified image
    '''
    args = OmegaConf.create(kw)
    OmegaConf.set_struct(args, True)
    pyscilog.log_to_file(args.outfile + '.log')
    pyscilog.enable_memory_logging(level=3)

    import numpy as np
    import xarray as xr
    import dask.array as da
    from glob import glob
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from PIL import Image

    print("Input options :")
    for key in kw.keys():
        print('     %25s = %s' % (key, args[key]), file=log)

    print("Reading data")
    Din = xr.open_dataset(args.data, engine='zarr', chunks={'time':1, 'nx':args.pix_chunks, 'ny':args.pix_chunks},)

    if args.tf == -1:
        args.tf = Din.time.size
    times = Din.times.data[args.t0:args.tf]
    rmss = Din.rmss.data[args.t0:args.tf]

    print('Slicing')
    image_out = Din.image.data[args.t0:args.tf,
                               args.x0:args.xf,
                               args.y0:args.yf].astype(np.float64)

    print('Saving')

    data_vars = {'image':(('time', 'nx', 'ny'), image_out)}
    coords = {'times': (('time',), times),
              'rmss': (('time',), rmss)}

    Dout = xr.Dataset(data_vars, coords)
    Dout.to_zarr(args.outfile + '.zarr', mode='w')

    imagep = image_out.compute()

    ntime = imagep.shape[0]
    times = times[args.t0:args.tf]

    print('2png')
    imgs = []
    for i in range(times.size):
        plt.figure()
        plt.imshow(imagep[i].T, cmap='inferno', norm=LogNorm(vmin=1e-6, vmax=0.15), origin='lower')
        plt.title(str(i))
        plt.savefig(args.outfile + str(i) + '.png', dpi=300)
        imgs.append(args.outfile + str(i) + '.png')
        plt.close()

    print('2gif')
    frames = []
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(args.outfile + '_raw.gif', format='GIF',
                   append_images=frames[1:],
                   save_all=True,
                   duration=args.duration*1000/ntime, loop=0)

