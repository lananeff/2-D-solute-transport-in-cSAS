import numpy as np

def us(x, y, alpha, A, eps):
    beta = alpha * (1 + 1j ) / np.sqrt(2)
    xi = (np.exp(beta) - 1) / (np.exp(beta) - np.exp(-beta))
    J = (1/ beta) *( (1 - xi) * (np.exp(beta) - 1) - xi * (np.exp(-beta) - 1) - beta)
    F = (1/beta) * ((1 - xi) * (np.exp(-beta * y) - 1) - xi * (np.exp(beta * y) - 1) + beta * y)
    f = xi * np.exp(beta * y) + (1 - xi) * np.exp(-beta * y) - 1
    df = beta*xi*np.exp(beta*y) - beta*(1 - xi) * np.exp(-beta * y) 
    vtild = -A/(2 * eps)
    mod_J = J * np.conjugate(J)
    

    t1 = 2*np.real(1j*df *np.conjugate(F))
    t2 = 2*np.real(1j * df / J)

    #us = vtild**2 * (x -1/2) * (t1/mod_J + y*t2)
    #us = (1/4) * (x -1/2) * (t1/mod_J + (A/eps)*y*t2) # Correct scaling - doesn't match old stokes drift
    us = (A/eps) * (1/4) * (x -1/2) * (t1/mod_J + y*t2) # Force to match old stokes drift for presentation

    return np.real(us)

def vs(x, y, alpha, A, eps):
    beta = alpha * (1 + 1j ) / np.sqrt(2)
    xi = (np.exp(beta) - 1) / (np.exp(beta) - np.exp(-beta))
    J = (1/ beta) *( (1 - xi) * (np.exp(beta) - 1) - xi * (np.exp(-beta) - 1) - beta)
    F = (1/beta) * ((1 - xi) * (np.exp(-beta * y) - 1) - xi * (np.exp(beta * y) - 1) + beta * y)
    f = xi * np.exp(beta * y) + (1 - xi) * np.exp(-beta * y) - 1
    vtild = -A/(2 * eps)
    t1 = 2*np.real(1j * F * np.conjugate(f))
    t2 = 2*np.real(1j* f / J)
    mod_J = J * np.conjugate(J)
    #vs = vtild**2 *( t1/ mod_J - y*t2)
    #vs = (1/(4)) *( t1/ mod_J - (A/eps) * y*t2) # Correct scaling - incorrect scaling of particle trajs
    vs = (A/eps) * (1/(4)) *( t1/ mod_J -   y*t2) # Force to match for presentation

    return  np.real(vs)