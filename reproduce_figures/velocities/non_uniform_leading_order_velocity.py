import numpy as np

def pno(x ,y ,t ,alpha ,A ,eps, forcing="sin", k=1):
    if forcing=="sin":
        def d(x):
            return np.sin(x*k*np.pi)
        def int_d(x):
            return -np.cos(x*k*np.pi)/(k*np.pi)
        def double_int_d(x):
            return -np.sin(x*k*np.pi)/(k**2 * np.pi**2)

    if forcing=="cos":
        def d(x):
            return np.cos(x*k*np.pi)
        def int_d(x):
            return np.sin(x*k*np.pi)/(k*np.pi)
        def double_int_d(x):
            return -np.cos(x*k*np.pi)/(k**2 * np.pi**2)

    if forcing=="uniform":
        def d(x):
            return 1
        def int_d(x):
            return x
        def double_int_d(x):
            return x**2/2


    

    beta = alpha * (1 + 1j ) / np.sqrt(2)
    xi = (np.exp(beta) - 1) / (np.exp(beta) - np.exp(-beta))
    J = (1/ beta) *( (1 - xi) * (np.exp(beta) - 1) - xi * (np.exp(-beta) - 1) - beta)
    f = xi * np.exp(beta * y) + (1 - xi) * np.exp(-beta * y) - 1
    coef = -beta**2  / (2*J)
    pterm = (double_int_d(x) + (double_int_d(0) - double_int_d(1) )*x - double_int_d(0))
    p0 = coef*pterm
    p = 2 * np.real(p0 * np.exp(1j * t)) 
    return p

def uno(x ,y ,t ,alpha ,A ,eps, forcing="sin", k=1):
    if forcing=="sin":
        def d(x):
            return np.sin(x*k*np.pi)
        def int_d(x):
            return -np.cos(x*k*np.pi)/(k*np.pi)
        def double_int_d(x):
            return -np.sin(x*k*np.pi)/(k**2 * np.pi**2)

    if forcing=="cos":
        def d(x):
            return np.cos(x*k*np.pi)
        def int_d(x):
            return np.sin(x*k*np.pi)/(k*np.pi)
        def double_int_d(x):
            return -np.cos(x*k*np.pi)/(k**2 * np.pi**2)

    if forcing=="uniform":
        def d(x):
            return 1
        def int_d(x):
            return x
        def double_int_d(x):
            return x**2/2


    

    beta = alpha * (1 + 1j ) / np.sqrt(2)
    xi = (np.exp(beta) - 1) / (np.exp(beta) - np.exp(-beta))
    J = (1/ beta) *( (1 - xi) * (np.exp(beta) - 1) - xi * (np.exp(-beta) - 1) - beta)
    f = xi * np.exp(beta * y) + (1 - xi) * np.exp(-beta * y) - 1
    px = beta**2 /(2*J) * (int_d(x) + (double_int_d(0) - double_int_d(1) ))
    u0 = -(px/beta**2)  * f
    u = 2 * np.real(u0 * np.exp(1j * t)) 
    return u

def vno(x ,y ,t ,alpha ,A ,eps, forcing="sin", k=1):
    if forcing=="sin":
        def d(x):
            return np.sin(x*k*np.pi)
        def int_d(x):
            return -np.cos(x*k*np.pi)/(k*np.pi)
        def double_int_d(x):
            return -np.sin(x*k*np.pi)/(k**2 * np.pi**2)

    if forcing=="cos":
        def d(x):
            return np.cos(x*k*np.pi)
        def int_d(x):
            return np.sin(x*k*np.pi)/(k*np.pi)
        def double_int_d(x):
            return -np.cos(x*k*np.pi)/(k**2 * np.pi**2)

    if forcing=="uniform":
        def d(x):
            return 1
        def int_d(x):
            return x
        def double_int_d(x):
            return x**2/2

    beta = alpha * (1 + 1j ) / np.sqrt(2)
    xi = (np.exp(beta) - 1) / (np.exp(beta) - np.exp(-beta))
    J = (1/ beta) *( (1 - xi) * (np.exp(beta) - 1) - xi * (np.exp(-beta) - 1) - beta)
    F = (1/beta) * ((1 - xi) * (np.exp(-beta * y) - 1) - xi * (np.exp(beta * y) - 1) + beta * y)
    v0 = -(d(x)/(2*J))*F
    v = 2 * np.real(v0 * np.exp(1j * t))

    return v