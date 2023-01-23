import numpy as np
import xarray as xr
from numba import jit
from scipy.optimize import fmin_l_bfgs_b as fmin
from datetime import datetime
from astropy.io import fits
from astropy.wcs import WCS
import aplpy
import matplotlib.pyplot as plt
import os.path
import nifty7 as ift


def to4d(data):
    if data.ndim == 4:
        return data
    elif data.ndim == 2:
        return data[None, None]
    elif data.ndim == 3:
        return data[None]
    elif data.ndim == 1:
        return data[None, None, None]
    else:
        raise ValueError("Only arrays with ndim <= 4 can be broadcast to 4D.")


def init_datetime(dtime):
    date, time = dtime.split('T')
    year, month, day = date.split('-')
    hour, minute, second = time.split(':')
    second, microsecond = second.split('.')
    return datetime(int(year), int(month), int(day), int(hour), int(minute), int(second)).timestamp()


def load_fits(name, dtype=np.float32):
    data = fits.getdata(name)
    data = np.transpose(to4d(data)[:, :, ::-1], axes=(0, 1, 3, 2))
    return np.require(data, dtype=dtype, requirements='C')


def save_fits(name, data, hdr, overwrite=True, dtype=np.float32):
    hdu = fits.PrimaryHDU(header=hdr)
    data = np.transpose(to4d(data), axes=(0, 1, 3, 2))[:, :, ::-1]
    hdu.data = np.require(data, dtype=dtype, requirements='F')
    hdu.writeto(name, overwrite=overwrite)


def set_wcs(cell_x, cell_y, nx, ny, radec, freq, t0, unit='Jy/beam'):
    """
    cell_x/y - cell sizes in degrees
    nx/y - number of x and y pixels
    radec - right ascention and declination in degrees
    freq - frequencies in Hz
    """

    w = WCS(naxis=4)
    w.wcs.ctype = ['RA---SIN', 'DEC--SIN', 'FREQ', 'STOKES']
    w.wcs.cdelt[0] = -cell_x
    w.wcs.cdelt[1] = cell_y
    w.wcs.cdelt[3] = 1
    w.wcs.cunit[0] = 'deg'
    w.wcs.cunit[1] = 'deg'
    w.wcs.cunit[2] = 'Hz'
    w.wcs.cunit[3] = unit
    if np.size(freq) > 1:
        ref_freq = freq[0]
    else:
        ref_freq = freq

    w.wcs.crval = [radec[0], radec[1], ref_freq, 1]
    w.wcs.crpix = [1 + nx//2, 1 + ny//2, 1, 1]

    if np.size(freq) > 1:
        w.wcs.cdelt[2] = freq[1]-freq[0]
        fmean = np.mean(freq)
    else:
        if isinstance(freq, np.ndarray):
            fmean = freq[0]
        else:
            fmean = freq

    header = w.to_header()
    header['RESTFRQ'] = fmean
    header['ORIGIN'] = 'Jovywood'
    header['BTYPE'] = 'Intensity'
    header['BUNIT'] = unit
    header['SPECSYS'] = 'TOPOCENT'
    t0 = np.round(t0)
    header['DATE-OBS'] = str(datetime.utcfromtimestamp(t0))

    return header


def abs_diff(x, xp):
    try:
        N, D = x.shape
        Np, D = xp.shape
    except Exception:
        N = x.size
        D = 1
        Np = xp.size
        x = np.reshape(x, (N, D))
        xp = np.reshape(xp, (Np, D))
    xD = np.zeros([D, N, Np])
    xpD = np.zeros([D, N, Np])
    for i in range(D):
        xD[i] = np.tile(x[:, i], (Np, 1)).T
        xpD[i] = np.tile(xp[:, i], (N, 1))
    return np.linalg.norm(xD - xpD, axis=0)


@jit(nopython=True, nogil=True, cache=True, inline='always')
def diag_dot(A, B):
    N = A.shape[0]
    C = np.zeros(N)
    for i in range(N):
        for j in range(N):
            C[i] += A[i, j] * B[j, i]
    return C


@jit(nopython=True, nogil=True, cache=True)
def dZdtheta(theta, xxsq, y, Sigma):
    '''
    Return log-marginal likelihood and derivs
    '''
    N = xxsq.shape[0]
    sigmaf = theta[0]
    l = theta[1]
    sigman = theta[2]

    # first the negloglik
    K = sigmaf**2 * np.exp(-xxsq/(2*l**2))
    Ky = K + np.diag(Sigma) * sigman**2
    # with numba.objmode: # args?
    #     u, s, v = np.linalg.svd(Ky, hermitian=True)
    u, s, v = np.linalg.svd(Ky)
    logdetK = np.sum(np.log(s))
    Kyinv = u.dot(v/s.reshape(N, 1))
    alpha = Kyinv.dot(y)
    Z = (np.vdot(y, alpha) + logdetK)/2

    # derivs
    dZ = np.zeros(theta.size)
    alpha = alpha.reshape(N, 1)
    aaT = Kyinv - alpha.dot(alpha.T)

    # deriv wrt sigmaf
    dK = 2 * K / sigmaf
    dZ[0] = np.sum(diag_dot(aaT, dK))/2

    # deriv wrt l
    dK = xxsq * K / l ** 3
    dZ[1] = np.sum(diag_dot(aaT, dK))/2

    # deriv wrt sigman
    dK = np.diag(2*sigman*Sigma)
    dZ[2] = np.sum(diag_dot(aaT, dK))/2

    return Z, dZ


def fit_pix(image, xxsq, Sigma, sigman0):
    return _fit_pix(image[0], xxsq, Sigma, sigman0)


@jit(nopython=True, nogil=True, cache=True)
def grid_search(sigmaf, sigman, xxsq, y, Sigma):
    Zmax = np.inf
    theta = np.array([sigmaf, 0.0, sigman])
    for l in np.arange(0.05, 0.65, 0.1):
        theta[1] = l
        Z, _ = dZdtheta(theta, xxsq, y, Sigma)
        if Z < Zmax:
            l0 = l
    return l0


def _fit_pix(image, xxsq, Sigma, sigman0):
    nt, nx, ny = image.shape
    thetas = np.zeros((3, nx, ny))
    for i in range(nx):
        for j in range(ny):
            y = np.ascontiguousarray(image[:, i, j])
            sigmaf0 = np.std(y)

            # l0 = grid_search(sigmaf0, sigman0, xxsq, y, Sigma)
            try:
                l0 = grid_search(sigmaf0, sigman0, xxsq, y, Sigma)
            except:
                l0 = 0.5

            theta0 = np.array([sigmaf0, l0, sigman0])
            try:
                theta, fval, dinfo = fmin(dZdtheta, theta0, args=(xxsq, y, Sigma), approx_grad=False,
                                          bounds=((1e-5, None), (1e-3, None), (0.1, 100)),
                                          factr=1e9, pgtol=5e-4)

                thetas[:, i, j] = theta
            except:
                thetas[:, i, j] = theta0

    return thetas


def interp_pix(thetas, image, xxsq, xxpsq, Sigma, oversmooth):
    return _interp_pix(thetas[0], np.require(image, dtype=np.float64), xxsq, xxpsq, Sigma, oversmooth)


@jit(nopython=True, nogil=True, cache=True, inline='always')
def _interp_pix(thetas, image, xxsq, xxpsq, Sigma, oversmooth):
    _, nx, ny = thetas.shape
    ntout = xxpsq.shape[0]
    imagep = np.zeros((ntout, nx, ny))
    for i in range(nx):
        for j in range(ny):
            y = image[:, i, j]
            theta = thetas[:, i, j]
            K = theta[0]**2*np.exp(-xxsq/(2*theta[1]**2))
            Ky = K + np.diag(Sigma) * oversmooth * theta[2]**2
            Kp = theta[0]**2*np.exp(-xxpsq/(2*theta[1]**2))
            imagep[:, i, j] = Kp.dot(np.linalg.solve(Ky, y))

    return imagep


def cube2fits(name, image, ras, decs, times, freqs, cell_size, idx):
    return _fitsmovie(name, image[0][0], ras, decs, times, freqs, cell_size, idx)


def _cube2fits(name, image, ras, decs, times, freqs, cell_size, idx):
    ntime, nx, ny = image.shape

    for i in range(ntime):
        radec = (ras[i], decs[i])
        hdr = set_wcs(cell_size, cell_size, nx, ny, radec, freqs, times[i])
        save_fits(f"{name}{idx[i]:04d}.fits", image[i], hdr, dtype=np.float32)

    return times

def madmask(data, wgt, th=5, sigv=7, sigt=7):
    import scipy
    from scipy.signal import convolve2d
    mask = None
    for i in range(4):
        image = np.where(wgt[i] > 0, np.abs(data[i]), 0)
        sig = scipy.stats.median_abs_deviation(image[image!=0], scale='normal')
        tmpmask = (image > th*sig)
        tmpmask = convolve2d(tmpmask, np.ones((sigv, sigt), dtype=np.float32), mode="same")
        tmpmask = (np.abs(tmpmask) > 0.1)
        if mask is None:
            mask = tmpmask
        else:
            mask = np.logical_or(mask, tmpmask)
    wgtmask = np.where(np.prod(wgt, axis=0) > 0, 0.0, 1.0)
    return np.logical_or(mask, wgtmask)

class SingleDomain(ift.LinearOperator):
    def __init__(self, domain, target):
        self._domain = ift.makeDomain(domain)
        self._target = ift.makeDomain(target)
        self._capability = self.TIMES | self.ADJOINT_TIMES

    def apply(self, x, mode):
        self._check_input(x, mode)
        return ift.makeField(self._tgt(mode), x.val)

# def drop2axes(filename, outname):
#     hdu = fits.open(filename)[0]
#     for kw in "CTYPE", "CRVAL", "CRPIX", "CDELT", "CUNIT":
#         for n in 3, 4:
#             try:
#                 hdu.header.remove(f"{kw}{n}")
#             except:
#                 pass
#     hdu.header['WCSAXES'] = 2
#     fits.writeto(outname, hdu.data[0,0]*1000, hdu.header, clobber=True)


# def _fits2png(idx, filenames):

#     for i in idx:
#         filename = filenames[i]
#         tmpname = filename + 'tmp.fits'
#         drop2axes(filename, tmpname)
#         fig0 = plt.figure(figsize=(20, 20))
#         fig = aplpy.FITSFigure(tmpname, subplot=[0,0,1,1], figure=fig0, dimensions=(0,1), slices=(0,0))
#         fig.show_colorscale(cmap='jet', vmin=2e-2, vmax=500, stretch='log')
#         fig.add_colorbar()
#         fig.colorbar.set_ticks([500, 100, 10, 1, 0.1])
#         fig.colorbar.set_axis_label_text('mJy/beam')
#         # fig.show_regions('dd.reg')
#         fig.add_label(0.99, 0.99, timestamps[filename], relative=True, color='yellow', size=20,
#                     horizontalalignment='right', verticalalignment='top')

#         xw, yw = fig.pixel2world(5000, 5000)
#         #fig.recenter(xw, yw, radius=0.25)

#         # Transient inset
#         fig1 = aplpy.FITSFigure(tmpname, subplot=[0.01,0.016,0.38,0.38], figure=fig0)
#         fig1.show_colorscale(cmap='jet', vmin=2e-2, vmax=500, stretch='log')
#         xw1 = (20 + 9/60 + 36.8/3600)*15
#         yw1 = -20 - 26/60 - 46/3600
#         fig1.recenter(xw, yw+0.07, radius=0.11)
#         # fig1.recenter(xw1, yw1, radius=0.11)
#         fig1.show_circles(xw1, yw1, 30*1/3600, edgecolors=['yellow'])
#         aplpy.AxisLabels(fig1).hide()
#         aplpy.Ticks(fig1).hide()
#         aplpy.TickLabels(fig1).hide()
#         aplpy.Frame(fig1).set_color('white')

#         # Callisto inset
#         tmpname = 'tmp.fits'
#         # filename1 = filename.replace("mean", "win5")
#         drop2axes(filename, tmpname)
#         fig2 = aplpy.FITSFigure(tmpname, subplot=[0.01,0.785,0.2,0.2], figure=fig0) # 0.785 0.771
#         #fig2.show_colorscale(cmap='nipy_spectral', vmin=5e-3, vmax=500, stretch='log')
#         fig2.show_colorscale(cmap='jet', vmin=1e-2, vmax=5, stretch='log')
#         xw2, yw2 = fig2.pixel2world(4601, 5101)
#         fig2.recenter(xw2, yw2, radius=0.12)
#         #xw2 = 15*(20 + 10/60+ + 03.59/3600)
#         #yw2 = -1*(20 + 34/60. + 03.3/3600)

#         # Callisto circle around starting position
#         #fig2.show_circles(xw2, yw2, 30*1/3600, edgecolors=['yellow'])
#         fig2.show_ellipses(xw2 - 25/3600., yw2, 2./60, 1./60, edgecolor='yellow')
#         #fig2.show_arrows(xw2 + 25/3600., yw2, 25/3600., 0, edgecolor='yellow')

#         aplpy.AxisLabels(fig2).hide()
#         aplpy.Ticks(fig2).hide()
#         aplpy.TickLabels(fig2).hide()
#         aplpy.Frame(fig2).set_color('white')

#         # Jove zoom
#         # filename3 = re.sub("im5/s..", "im5/cubes", filename.replace("MFS", "cube"))
#         fig3 = aplpy.FITSFigure(filename, subplot=[0.68,0.016,0.3,0.3], figure=fig0,
#                                 slices=[7,0])
#         fig3._wcs = fig3._wcs.dropaxis(3).dropaxis(2)
#         fig3.show_colorscale(cmap='jet', vmin=5e-4, vmax=0.15, stretch='log')
#         fig3.recenter(xw, yw, radius=0.025)
#         aplpy.AxisLabels(fig3).hide()
#         aplpy.Ticks(fig3).hide()
#         aplpy.TickLabels(fig3).hide()
#         aplpy.Frame(fig3).set_color('white')


#         fig0.savefig('output/mfs/png/' + os.path.splitext(os.path.basename(filename))[0] + ".png",
#                     facecolor='white', edgecolor='none', transparent=False, bbox_inches='tight')

#         plt.close(fig0)