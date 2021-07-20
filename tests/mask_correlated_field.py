import numpy as np
import nifty7 as ift

def interp2d():
    wgt = np.ones((128, 256), dtype=np.float64)
    data = np.ones((128, 256), dtype=np.float64)

    # signal domain
    npix1, npix2 = wgt.shape
    position_space = ift.RGSpace([npix1, npix2])

    args = {
        'offset_mean': 0,
        'offset_std': (1e-3, 1e-6),

        # Amplitude of field fluctuations
        'fluctuations': (1., 0.8),  # 1.0, 1e-2

        # Exponent of power law power spectrum component
        'loglogavgslope': (-3., 1),  # -6.0, 1

        # Amplitude of integrated Wiener process power spectrum component
        'flexibility': (2, 1.),  # 1.0, 0.5

        # How ragged the integrated Wiener process component is
        'asperity': (0.5, 0.4)  # 0.1, 0.5
    }

    correlated_field = ift.SimpleCorrelatedField(position_space, **args)

    mask = np.where(wgt > 0, 0.0, 1.0)
    mask = ift.makeField(correlated_field.domain, mask)


if __name__=="__main__":
    interp2d()