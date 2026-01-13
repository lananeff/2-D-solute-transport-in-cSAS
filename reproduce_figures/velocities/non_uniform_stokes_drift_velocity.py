import numpy as np

def usno(x, y, alpha ,A ,eps, forcing, k=1):
    if forcing=="sin":
        def d(x):
            return np.sin(x*k*np.pi)
        def dx(x):
            return k*np.pi*np.cos(x*k*np.pi)
        def int_d(x):
            return -np.cos(x*k*np.pi)/(k*np.pi)
        def double_int_d(x):
            return -np.sin(x*k*np.pi)/(k**2 * np.pi**2)

    if forcing=="cos":
        def d(x):
            return np.cos(x*k*np.pi)
        def dx(x):
            return -k*np.pi * np.sin(k*np.pi * x)
        def int_d(x):
            return np.sin(x*k*np.pi)/(k*np.pi)
        def double_int_d(x):
            return -np.cos(x*k*np.pi)/(k**2 * np.pi**2)

    if forcing=="uniform":
        def d(x):
            return 1
        def dx(x):
            return 0
        def int_d(x):
            return x
        def double_int_d(x):
            return x**2/2

    beta = alpha * (1 + 1j ) / np.sqrt(2)
    xi = (np.exp(beta) - 1) / (np.exp(beta) - np.exp(-beta))
    J = (1/ beta) *( (1 - xi) * (np.exp(beta) - 1) - xi * (np.exp(-beta) - 1) - beta)
    Jc = np.conjugate(J)
    f = xi * np.exp(beta * y) + (1 - xi) * np.exp(-beta * y) - 1
    delta = double_int_d(0) - double_int_d(1)
    F = (1/beta) * ((1 - xi) * (np.exp(-beta * y) - 1) - xi * (np.exp(beta * y) - 1) + beta * y)
    df = xi* beta * np.exp(beta * y) -beta * (1-xi) * np.exp(-beta * y)
    mod_J = J * np.conj(J)

    us1 = 0
    us2 = 2*np.real( (1j * d(x) * df * (int_d(x) + delta) * np.conj(F)) / (4 * mod_J))
    us3 = 2 * np.real((1j * y * d(x) * (int_d(x) + delta) * df ) / (4 * J))

    us = us1 + us2 + us3
    return (A/eps) * us


def vsno(x, y, alpha ,A ,eps, forcing, k=1):
    if forcing=="sin":
        def d(x):
            return np.sin(x*k*np.pi)
        def dx(x):
            return k*np.pi*np.cos(x*k*np.pi)
        def int_d(x):
            return -np.cos(x*k*np.pi)/(k*np.pi)
        def double_int_d(x):
            return -np.sin(x*k*np.pi)/(k**2 * np.pi**2)

    if forcing=="cos":
        def d(x):
            return np.cos(x*k*np.pi)
        def dx(x):
            return -k*np.pi * np.sin(k*np.pi * x)
        def int_d(x):
            return np.sin(x*k*np.pi)/(k*np.pi)
        def double_int_d(x):
            return -np.cos(x*k*np.pi)/(k**2 * np.pi**2)

    if forcing=="uniform":
        def d(x):
            return 1
        def dx(x):
            return 0
        def int_d(x):
            return x
        def double_int_d(x):
            return x**2/2
    
    beta = alpha * (1 + 1j ) / np.sqrt(2)
    xi = (np.exp(beta) - 1) / (np.exp(beta) - np.exp(-beta))
    J = (1/ beta) *( (1 - xi) * (np.exp(beta) - 1) - xi * (np.exp(-beta) - 1) - beta)
    f = xi * np.exp(beta * y) + (1 - xi) * np.exp(-beta * y) - 1
    delta = double_int_d(0) - double_int_d(1)
    F = (1/beta) * ((1 - xi) * (np.exp(-beta * y) - 1) - xi * (np.exp(beta * y) - 1) + beta * y)
    df = xi* beta * np.exp(beta * y) -beta * (1-xi) * np.exp(-beta * y)
    mod_J = J * np.conj(J)

    vs1 = 2*np.real((-1j * dx(x) * (int_d(x) + delta) * f * np.conj(F))/(4 * mod_J))
    #vs2 = 2*np.real((-1j * d(x)**2 *f * np.conj(F))/ (4 * mod_J) )
    vs2 = 2*np.real((1j * d(x)**2 * F * np.conj(f))/(4 * mod_J))
    vs3 = 2*np.real(( 1j * d(x)**2 * y * ( np.conj(f))) / (4 * np.conj(J)))

    vs4 = 0*2*np.real((1j * dx(x) * (int_d(x) + delta) * f * y)/(4*J) )

    vs = (vs1 + vs2 +vs3 + vs4)

    return (A/eps) * vs

