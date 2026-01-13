import numpy as np

# Default d(x) function (sin(pi * x))
def default_d(x):
    d = np.sin(x*np.pi)
    int_d = -np.cos(x*np.pi)/np.pi
    double_int_d = -np.sin(x*np.pi)/(np.pi**2)
    return [d, int_d, double_int_d]


# Another example d(x) function (cos(pi * x))
def cosine_d(x):
    return np.cos(np.pi * x)