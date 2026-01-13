import numpy as np

def u0(x ,y ,t ,alpha ,A ,eps):
    beta = alpha * (1 + 1j ) / np.sqrt(2)
    xi = (np.exp(beta) - 1) / (np.exp(beta) - np.exp(-beta))
    J = (1/ beta) *( (1 - xi) * (np.exp(beta) - 1) - xi * (np.exp(-beta) - 1) - beta)
    f = xi * np.exp(beta * y) + (1 - xi) * np.exp(-beta * y) - 1
    u0 = -1 /(4 * J) * (2*x - 1) * f
    u = 2 * np.real(u0 * np.exp(1j * t)) #removed A/eps
    return u

def v0(x ,y ,t ,alpha ,A ,eps):
    beta = alpha * (1 + 1j ) / np.sqrt(2)
    xi = (np.exp(beta) - 1) / (np.exp(beta) - np.exp(-beta))
    J = (1/ beta) *( (1 - xi) * (np.exp(beta) - 1) - xi * (np.exp(-beta) - 1) - beta)
    F = (1/beta) * ((1 - xi) * (np.exp(-beta * y) - 1) - xi * (np.exp(beta * y) - 1) + beta * y)
    v0 = -1/(2 * J) * F
    v = 2 * np.real(v0 * np.exp(1j * t)) # removed A/eps

    return v
