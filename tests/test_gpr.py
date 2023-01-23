import numpy as np

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

    K = theta[0]**2*np.exp(-xxsq/(2*theta[1]**2))
    Ky = K + np.diag(Sigma) * 2 * theta[2]**2
    xxpsq = abs_diff(tp, t)
    Kp = theta[0]**2*np.exp(-xxpsq/(2*theta[1]**2))
    fp = Kp.dot(np.linalg.solve(Ky, y))

    import matplotlib.pyplot as plt

    plt.plot(t, f, 'k')
    plt.plot(tp, fp, 'b')
    plt.errorbar(t, y, np.sqrt(Sigma)*theta[2], fmt='xr')
    plt.show()