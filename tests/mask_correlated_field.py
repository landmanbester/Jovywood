import numpy as np
import nifty7 as ift

def interp2d(wgt, data):
    """
    We are given

    wgt - (ntime, nchan) array of weights (float)
    data - (ntime, nchan) array of data (float)

    and we want to interpolate and smooth data to reconstruct a underlying GRF.
    """

    # signal domain
    npix1, npix2 = wgt.shape
    position_space = ift.RGSpace([npix1, npix2])

    # harmonic space and harmonic transform
    hspace = position_space.get_default_codomain()
    HT = ift.HarmonicTransformOperator(hspace, position_space)

    # Domain on which the field's degrees of freedom are defined
    domain = ift.DomainTuple.make(hspace)

    # we mask regions with zero weights
    mask = np.where(wgt > 0, 0.0, 1.0)
    mask = ift.Field.from_raw(position_space, mask)
    Mask = ift.MaskOperator(mask)

    # Define amplitude (square root of power spectrum)
    def power_spectrum(k):
        return 10./(10. + k**2)

    # 1D spectral space on which the power spectrum is defined
    power_space = ift.PowerSpace(hspace)

    # Mapping to (higher dimensional) harmonic space
    PD = ift.PowerDistributor(hspace, power_space)

    # Apply the mapping
    prior_correlation_structure = PD(ift.PS_field(power_space, power_spectrum))

    # Insert the result into the diagonal of an harmonic space operator
    S = ift.DiagonalOperator(prior_correlation_structure)

    # measurement operator
    R = Mask @ HT

    data_space = R.target

    # set variances
    var = 1.0/wgt[wgt>0]
    N = ift.DiagonalOperator(ift.makeField(data_space, var))

    data = ift.makeField(data_space, data[wgt > 0])

    ##### WIENER FILTER solution #####
    D_inv = R.adjoint @ N.inverse @ R + S.inverse
    j = R.adjoint_times(N.inverse_times(data))
    IC = ift.GradientNormController(iteration_limit=500, tol_abs_gradnorm=1e-1)
    D = ift.InversionEnabler(D_inv, IC, approximation=S.inverse).inverse
    m = D(j)

    #### MAP soln + samples with known power spectrum #####
    a = ift.PS_field(power_space, sqrtpspec)
    A = ift.makeOp(PD(a))
    R = Mask @ HT @ A

    IC = ift.DeltaEnergyController(
        name='Newton', iteration_limit=100, tol_rel_deltaE=1e-4)
    likelihood_energy = ift.GaussianEnergy(mean=data, inverse_covariance=N.inverse) @ R
    Ham = ift.StandardHamiltonian(likelihood_energy, IC)
    initial_position = ift.from_random(domain, 'normal')
    H = ift.EnergyAdapter(initial_position, Ham, want_metric=True)

    # Minimize
    minimizer = ift.NewtonCG(IC)
    H, _ = minimizer(H)

    # Draw posterior samples
    N_samples = 5
    metric = Ham(ift.Linearization.make_var(H.position, want_metric=True)).metric
    samples = [metric.draw_sample(from_inverse=True) + H.position
               for _ in range(N_samples)]

    sc = ift.StatCalculator()
    for s in samples:
        sc.add(HT(A(s)))

    ##### but how to do it with a correlated field? #####
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
    pspec = correlated_field.power_spectrum

    # no non-linearity so signal is correlated field
    signal = correlated_field

    # I want to define a response that just acts on the image
    # while leaving the rest of the correlated field in tact.
    # Something like
    mask = np.where(wgt > 0, 0.0, 1.0)
    mask = ift.makeField(signal.domain, mask)
    Mask = ift.MaskOperator(mask)

    # ???
