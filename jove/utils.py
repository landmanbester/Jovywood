import numpy as np
import xarray as xr
from numba import jit
from scipy.optimize import fmin_l_bfgs_b as fmin
from datetime import datetime
from astropy.io import fits
from astropy.wcs import WCS


def to5d(data):
    if data.ndim == 5:
        return data
    elif data.ndim == 4:
        return data[None]
    elif data.ndim == 3:
        return data[None, None]
    elif data.ndim == 2:
        return data[None, None, None]
    elif data.ndim == 1:
        return data[None, None, None, None]
    else:
        raise ValueError("Only arrays with ndim <= 5 can be broadcast to 5D.")


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


def save_fits(name, data, hdr, overwrite=True, dtype=np.float32, ndim=4):
    hdu = fits.PrimaryHDU(header=hdr)
    if ndim == 4:
        data = np.transpose(to4d(data), axes=(0, 1, 3, 2))[:, :, ::-1]
    elif ndim == 5:
        data = np.transpose(to4d(data), axes=(0, 1, 2, 4, 3))[:, :, :, ::-1]
    hdu.data = np.require(data, dtype=dtype, requirements='F')
    hdu.writeto(name, overwrite=overwrite)


def set_wcs(cell_x, cell_y, nx, ny, radec, freq, time, unit='Jy/beam'):
    """
    cell_x/y - cell sizes in degrees
    nx/y - number of x and y pixels
    radec - right ascention and declination in degrees
    freq - frequencies in Hz
    """

    w = WCS(naxis=5)
    w.wcs.ctype = ['RA---SIN', 'DEC--SIN', 'FREQ', 'STOKES', 'TIME']
    w.wcs.cdelt[0] = -cell_x
    w.wcs.cdelt[1] = cell_y
    w.wcs.cdelt[3] = 1
    w.wcs.cunit[0] = 'deg'
    w.wcs.cunit[1] = 'deg'
    w.wcs.cunit[2] = 'Hz'
    if np.size(freq) > 1:
        ref_freq = freq[0]
    else:
        ref_freq = freq
    if np.size(time) > 1:
        ref_time = time[0]
    else:
        ref_time = time

    w.wcs.crval = [radec[0], radec[1], ref_freq, 1, ref_time]
    w.wcs.crpix = [1 + nx//2, 1 + ny//2, 1, 1, 1]

    if np.size(freq) > 1:
        w.wcs.cdelt[2] = freq[1]-freq[0]
        fmean = np.mean(freq)
    else:
        if isinstance(freq, np.ndarray):
            fmean = freq[0]
        else:
            fmean = freq

    if np.size(time) > 1:
        w.wcs.cdelt[4] = time[1]-time[0]

    header = w.to_header()
    header['RESTFRQ'] = fmean
    header['ORIGIN'] = 'Jovywood'
    header['BTYPE'] = 'Intensity'
    header['BUNIT'] = unit
    header['SPECSYS'] = 'TOPOCENT'

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


def _fit_pix(image, xxsq, Sigma, sigman0):
    nt, nx, ny = image.shape
    thetas = np.zeros((3, nx, ny))
    for i in range(nx):
        for j in range(ny):
            y = np.ascontiguousarray(image[:, i, j])
            theta0 = np.array([np.std(y), 0.025, sigman0])

            try:
                theta, fval, dinfo = fmin(dZdtheta, theta0, args=(xxsq, y, Sigma), approx_grad=False,
                                          bounds=((1e-5, None), (1e-4, None), (0.1, 100)),
                                          factr=1e7, pgtol=1e-4)

                thetas[:, i, j] = theta
            except:
                thetas[:, i, j] = np.ones(3)

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


def fitsmovie(name, image, ras, decs, times, freqs, cell_size, idx):
    return _fitsmovie(name, image[0][0], ras, decs, times, freqs, cell_size, idx)


def _fitsmovie(name, image, ras, decs, times, freqs, cell_size, idx):
    ntime, nx, ny = image.shape

    for i in range(ntime):
        radec = (ras[i], decs[i])
        hdr = set_wcs(cell_size, cell_size, nx, ny, radec, freqs, times[i])
        save_fits(name + str(idx[i]) + '.fits', image[i], hdr, dtype=np.float32, ndim=5)

    return times