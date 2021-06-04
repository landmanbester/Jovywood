
import click
from omegaconf import OmegaConf
import pyscilog
pyscilog.init('jove')
log = pyscilog.get_logger('CONCAT')

@cli.command()
@click.option("-d", "--data", type=str,
              help="Path to data.zarr")
@click.option("-h", "--hypers", type=str,
              help="Path to hypers.zarr")
@click.option("-o", "--outfile", type=str, default='pfb',
              help='Base name of output file.')
@click.option("-pc", "--pix_chunks", type=int, default=50,
              help='Pixel chunks')
@click.option('-nto', "--ntime-out", type=int,
              help="Number of output times")
@click.option('-dur', "--duration", type=float, default=5,
              help="Duration of gif in ms")
@click.option('-os', "--oversmooth", type=int, default=5,
              help="Noise inflation factor")
@click.option('-nthreads', '--nthreads', type=int, default=64,
              help='Number of dask threads.')
def interp(**kw):
    args = OmegaConf.create(kw)
    OmegaConf.set_struct(args, True)
    pyscilog.log_to_file(args.output_filename + '.log')
    pyscilog.enable_memory_logging(level=3)

    print("Input options :")
    for key in GD.keys():
        print(key + ' = ', args[key])

    import os
    os.environ["OMP_NUM_THREADS"] = str(1)
    os.environ["OPENBLAS_NUM_THREADS"] = str(1)
    os.environ["MKL_NUM_THREADS"] = str(1)
    os.environ["VECLIB_MAXIMUM_THREADS"] = str(1)
    os.environ["NUMBA_NUM_THREADS"] = str(1)
    import numpy as np
    import xarray as xr
    from africanus.gps.utils import abs_diff
    import dask.array as da
    import dask
    from dask.diagnostics import ProgressBar
    from PIL import Image
    from glob import glob
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    from jove.utils import interp_pix

    from multiprocessing.pool import Pool
    with Pool(processes=args.nthreads) as pool:
        dask.config.set(pool=pool)

        Din = xr.open_dataset(args.data, chunks={'time':-1,
                                                'nx':args.pix_chunks,
                                                'ny':args.pix_chunks},
                                                engine='zarr')

        image = Din.image.data
        rmss = Din.rmss.data
        times = Din.times.data
        ras = Din.ras.data
        decs = Din.decs.data

        thetas = xr.open_dataset(args.hypers, chunks={'time':-1,
                                                      'nx':args.pix_chunks,
                                                      'ny':args.pix_chunks},
                                                      engine='zarr').thetas.data

        ntime, nx, ny = image.shape

        # normalise to between 0 and 1
        t = times - times.min()
        t = t/t.max()

        # precompute abs diffs
        xxsq = abs_diff(t, t)
        tp = np.linspace(0, 1, args.ntime_out)
        xxpsq = abs_diff(tp, t)

        Sigma = rmss**2
        sigman0 = 1.0

        thetas = da.blockwise(
            interp_pix, 'txy',
            thetas, 'pxy',
            image, 'txy',
            xxsq, None,
            xxpsq, None,
            Sigma, None,
            args.oversmooth, None,
            align_arrays=False,
            dtype=image.dtype
        )

        print('Fitting')
        with ProgressBar():
            thetas = dask.compute(thetas)[0]  # scheduler='processes'

        image_out = da.from_array(thetas, chunks = (3, args.pix_chunks, args.pix_chunks), name=False)

        data_vars = {'theta':(('three', 'nx', 'ny'), image_out)}
        coords = {'times': (('time',), tp)}

        Dout = xr.Dataset(data_vars, coords)
        Dout.to_zarr(args.outfile + '_hypers.zarr', mode='w')

    print('2png')
    imgs = []
    for i in range(args.ntime_out):
        plt.figure()
        plt.imshow(imagep[i].T, cmap='inferno', vmin=1e-6, vmax=0.15, origin='lower')
        plt.title(str(i))
        plt.savefig(args.outfile + str(i) + '.png', dpi=300)
        imgs.append(args.outfile + str(i) + '.png')
        plt.close()

    print('2gif')
    frames = []
    for i in imgs:
        new_frame = Image.open(i)
        frames.append(new_frame)

    frames[0].save(args.outfile + '.gif', format='GIF',
                append_images=frames[1:],
                save_all=True,
                duration=args.duration*1000/args.ntime_out, loop=0)




def test_gpr():
    N = 100
    t = np.sort(np.random.random(N))
    Np = 250
    tp = np.linspace(0, 1, Np)
    f = np.cos(10*t)
    Sigma = 1e-3 * (1 + np.random.random(N))
    y = f + np.sqrt(Sigma) * np.random.randn(N)

    sigmaf0 = np.std(y)
    theta0 = np.array([np.std(y), 1.0, 1.0])


    xxsq = abs_diff(t, t)


    from time import time
    ti = time()
    theta, fval, dinfo = fmin(dZdtheta, theta0, args=(xxsq, y, Sigma), approx_grad=False,
                              bounds=((1e-3, None), (1e-3, None), (1e-3, None)))
                              # factr=1e9, pgtol=1e-4)
    print(time() - ti)
    print(theta)
    print(fval)
    print(dinfo)

    # theta[1] = 2*theta[1]

    # K = theta[0]**2*np.exp(-xxsq/(2*theta[1]**2))
    # Ky = K + np.diag(Sigma) * 2 * theta[2]**2
    # xxpsq = abs_diff(tp, t)
    # Kp = theta[0]**2*np.exp(-xxpsq/(2*theta[1]**2))
    # fp = Kp.dot(np.linalg.solve(Ky, y))

    # import matplotlib.pyplot as plt

    # plt.plot(t, f, 'k')
    # plt.plot(tp, fp, 'b')
    # plt.errorbar(t, y, np.sqrt(Sigma)*theta[2], fmt='xr')
    # plt.show()



# 24 * 5 time stamps