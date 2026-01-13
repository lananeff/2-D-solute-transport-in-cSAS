import numpy as np
from scipy.integrate import quad
from .steady_streaming_velocity import B11, newT, B1, newR, T0

# Define the sigmoid-based q(x)
def q(x, q0=1, center=0.6, steepness=100):
    return  q0*(1 - 1 / (1 + np.exp(-steepness * (x - center))) + ( - 1 / (1 + np.exp(-steepness * (1 - x - center)))))


def int_q(x, q0=1, center=0.6, steepness=100):
    u = np.exp(-steepness * (1 - x- center))
    v = np.exp(-steepness * (x - center))
    return q0*(x+ (1/steepness) * (np.log(v) - np.log((1+v)*steepness) -np.log(u) + np.log((1+u)*steepness)))


def Q(x, q0=1, center=0.6, steepness=100):
    numerator = np.exp(steepness) + np.exp(2 * center * steepness) + 2 * np.exp(steepness * (center + x))
    denominator = np.exp(steepness) - np.exp(2 * center * steepness)
    return (2 * q0 * np.arctanh(numerator / denominator)) / steepness


def upd(x, y, alpha, A, eps, omega, h, center=0.6, steepness=30): 
    q0 = 4*9.5e-6 / (6 * A* omega * h * (center - 1/2) ) # Same as tilde{q_0} / (A omega eps)
    def fint(x):
        # define integral of q
        q1 = np.exp(-steepness * (1 - x- center))
        q2 = np.exp(-steepness * (x - center))
        Q = (x+ (1/steepness) * (np.log(q2) - np.log((1+q2)*steepness) -np.log(q1) + np.log((1+q1)*steepness)))
        return Q
    
    upd = 6*q0*(fint(x)-fint(1/2))*(y**2 + y)
    
    return (0.5) * (upd + np.conjugate(upd))

def vpd(x, y, alpha, A, eps, omega, h, center = 0.6, steepness = 30):
    q0 = 4*9.5e-6 / (6 * A* omega * h * (center - 1/2))
    def fint(x):
        # define integral of q
        q1 = np.exp(-steepness * (1 - x- center))
        q2 = np.exp(-steepness * (x - center))
        Q = (x+ (1/steepness) * (np.log(q2) - np.log((1+q2)*steepness) -np.log(q1) + np.log((1+q1)*steepness)))
        return Q

    qval = q0*(1 - 1 / (1 + np.exp(-steepness * (x - center))) + ( - 1 / (1 + np.exp(-steepness * (1 - x - center)))))
    vpd = qval * (1 - 6*(y**3/3 + y**2/2))

    return (0.5) * (vpd + np.conjugate(vpd))