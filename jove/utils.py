

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


def set_wcs(cell_x, cell_y, nx, ny, radec, freq, unit='Jy/beam'):
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
    if np.size(freq) > 1:
        ref_freq = freq[0]
    else:
        ref_freq = freq
    w.wcs.crval = [radec[0]*180.0/np.pi, radec[1]*180.0/np.pi, ref_freq, 1]
    w.wcs.crpix = [1 + nx//2, 1 + ny//2, 1, 1]

    if np.size(freq) > 1:
        w.wcs.crval[2] = freq[0]
        df = freq[1]-freq[0]
        w.wcs.cdelt[2] = df
        fmean = np.mean(freq)
    else:
        if isinstance(freq, np.ndarray):
            fmean = freq[0]
        else:
            fmean = freq

    header = w.to_header()
    header['RESTFRQ'] = fmean
    header['ORIGIN'] = 'pfb-clean'
    header['BTYPE'] = 'Intensity'
    header['BUNIT'] = unit
    header['SPECSYS'] = 'TOPOCENT'

    return header
