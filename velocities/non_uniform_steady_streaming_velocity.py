import numpy as np


def load_default_forcing(k):
    """
    Load default (sinusoidal) gamma, Gamma, and G functions.
    
    Returns:
    - dict of callable functions for gamma1, gamma2, gamma3, Gamma1, Gamma2, Gamma3, Gamma4, Gamma5, G4, G5
    """
    def d0(x, k):
        return np.sin(k* np.pi * x)

    def D(x, k):
        return - np.cos(k* np.pi * x) / (k* np.pi)

    def D2(x, k):
        return -np.sin(k * np.pi * x) / (k**2 * np.pi**2)

    delta = D2(0,k) - D2(1, k)

        
    def gamma(x, k):
        kp = k * np.pi
        return (-np.cos(kp * x) / kp + np.sin(kp) / (k**2 * np.pi**2)) * np.sin(kp * x)

    def gammax(x, k):
        kp = k * np.pi
        return -np.cos(2 * kp * x) + (np.cos(kp * x) * np.sin(kp)) / kp

    def gamma_int(x, k):
        kp = k * np.pi
        numerator = (- kp * np.cos(kp * x) + np.sin(kp))**2
        denominator = 2 * k**4 * np.pi**4
        return numerator / denominator

    
    def g1x(x, k):
        return np.cos(k*np.pi*x) * (-np.cos(k*np.pi*x) + np.sin(k*np.pi)/(k*np.pi))

    def g2x(x, k):
        return np.sin(k* np.pi * x) **2


    def g1(x, k):
        kp = k*np.pi
        num = 4*np.sin(kp)*np.sin(kp*x) - kp*(2*kp*x + np.sin(2*kp*x))
        den = 4*kp**2
        with np.errstate(divide='ignore', invalid='ignore'):
            val = num/den
        return 0.0 if kp == 0 else val

    def g2(x, k):
        return 0.5 * x - np.sin(2 * k * np.pi * x) / (4 * k * np.pi)

    def G1(x, k):
        kp = k * np.pi
        return (-2*(kp**3)*(np.asarray(x))**2 + kp*np.cos(2*kp*np.asarray(x)) - 8*np.cos(kp*np.asarray(x))*np.sin(kp)) / (8*(kp**3))


    def G2(x, k):
        return 0.125 * (2 * x**2 + np.cos(2 * k * np.pi * x) / (k**2 * np.pi**2))

    functions = {
        "gamma_x": lambda x: gammax(x, k),
        "gamma": lambda x: gamma(x, k),
        "Gamma": lambda x: gamma_int(x, k),

        "g1_x": lambda x: g1x(x, k),
        "g2_x": lambda x: g2x(x, k),
        "g1": lambda x: g1(x, k),
        "g2": lambda x: g2(x, k),

        "G1": lambda x: G1(x, k),
        "G2": lambda x: G2(x, k),
    }
    
    return functions

def cos_forcing(k):
    """
    Load default (sinusoidal) gamma, Gamma, and G functions.
    
    Returns:
    - dict of callable functions for gamma1, gamma2, gamma3, Gamma1, Gamma2, Gamma3, Gamma4, Gamma5, G4, G5
    """
    def d0(x, k):
        return np.cos(k* np.pi * x)

    def D(x, k):
        return np.sin(k* np.pi * x) / (k* np.pi)

    def D2(x, k):
        return -np.cos(k * np.pi * x) / (k**2 * np.pi**2)

    delta = D2(0,k) - D2(1, k)

        
    def gamma(x, k):
        kp = k * np.pi
        return np.cos(kp * x) * (-(1 / (k**2 * np.pi**2)) + np.cos(kp) / (k**2 * np.pi**2) + np.sin(kp * x) / kp)

    def gammax(x, k):
        kp = k * np.pi
        return np.cos(2 * kp * x) + (2 * np.sin(kp / 2)**2 * np.sin(kp * x)) / kp

    def gamma_int(x, k):
        kp = k * np.pi
        return (np.sin(kp * x) * (-2 + 2 * np.cos(kp) + kp * np.sin(kp * x))) / (2 * (k**3) * (np.pi**3))

    
    def g1x(x, k):
        kp = k * np.pi
        return -(np.sin(kp * x) * (-1 + np.cos(kp) + kp * np.sin(kp * x))) / kp

    def g2x(x, k):
        kp = k * np.pi
        return np.cos(kp * x)**2


    def g1(x, k):
        kp = k * np.pi
        return (4 * (-1 + np.cos(kp)) * np.cos(kp * x) + kp * (-2 * kp * x + np.sin(2 * kp * x))) / (4 * (k**2) * (np.pi**2))

    def g2(x, k):
        kp = k * np.pi
        return x/2 + np.sin(2 * kp * x) / (4 * kp)

    def G1(x, k):
        kp = k * np.pi
        return 0.125 * (-2 * x**2 + (-kp * np.cos(2 * kp * x) + 8 * (-1 + np.cos(kp)) * np.sin(kp * x)) / (k**3 * np.pi**3))

    def G2(x, k):
        kp = k * np.pi
        return 0.25 * (x**2 - np.cos(2 * kp * x) / (2 * (k**2) * (np.pi**2)))

    functions = {
        "gamma_x": lambda x: gammax(x, k),
        "gamma": lambda x: gamma(x, k),
        "Gamma": lambda x: gamma_int(x, k),

        "g1_x": lambda x: g1x(x, k),
        "g2_x": lambda x: g2x(x, k),
        "g1": lambda x: g1(x, k),
        "g2": lambda x: g2(x, k),

        "G1": lambda x: G1(x, k),
        "G2": lambda x: G2(x, k),
    }
    
    return functions

def uniform_forcing():

    def d0(x):
        return 1

    def D(x):
        return x

    def D2(x):
        return x**2 / 2

    delta = D2(0) - D2(1)

    def gamma(x):
        return x-1/2

    def gammax(x):
        return 1

    def gamma_int(x):
        return (x**2 - x)/2

    def g1x(x):
        return 0

    def g2x(x):
        return 1

    def g1(x):
        return 0

    def g2(x):
        return x

    def G1(x):
        return 0

    def G2(x):
        return x**2/2

    functions = {
        "gamma_x": lambda x: gammax(x),
        "gamma": lambda x: gamma(x),
        "Gamma": lambda x: gamma_int(x),

        "g1_x": lambda x: g1x(x),
        "g2_x": lambda x: g2x(x),
        "g1": lambda x: g1(x),
        "g2": lambda x: g2(x),

        "G1": lambda x: G1(x),
        "G2": lambda x: G2(x),
    }
    
    return functions

def J(alpha):
    beta = alpha * (1 + 1j ) / np.sqrt(2)
    xi = (np.exp(beta) - 1) / (np.exp(beta) - np.exp(-beta))
    J = (1/ beta) *( (1 - xi) * (np.exp(beta) - 1) - xi * (np.exp(-beta) - 1) - beta)
    return J

def kappa(alpha, A, eps):
    return (A * alpha**2)/ (J(alpha)* np.conjugate(J(alpha)) * eps * 4)

def T1(y, B, alpha):
    denominator = 4 * B**2 * alpha**2 * (np.cos(B * alpha) + np.cosh(B * alpha))
    term1 = 2 * B**2 * y**2 * alpha**2 * np.cos(B * alpha)
    term2 = 2j * (np.cos(B * (1 + (1 - 1j) * y) * alpha) - np.cos(B * (1 + (1 + 1j) * y) * alpha))
    term3 = -np.cos(B * (1 + 2 * y) * alpha)
    term4 = 2 * B**2 * y**2 * alpha**2 * np.cosh(B * alpha)
    term5 = -2j * np.cosh(B * (1 + (1 - 1j) * y) * alpha)
    term6 = 2j * np.cosh(B * (1 + (1 + 1j) * y) * alpha)
    term7 = np.cosh(B * (1 + 2 * y) * alpha)
    return (term1 + term2 + term3 + term4 + term5 + term6 + term7) / denominator

def T2(y, B, alpha):
    denominator = 4 * B**2 * alpha**2 * (np.cos(B * alpha) + np.cosh(B * alpha))
    expr = (-4 - 2j) * np.cos(B * (1 + (1 - 1j) * y) * alpha) + \
           np.cos(B * (1 + 2 * y) * alpha) - \
           (4 - 2j) * np.cosh(B * (1 + (1 + 1j) * y) * alpha) + \
           np.cosh(B * (1 + 2 * y) * alpha) - \
           (2 - 2j) * B * y * alpha * np.sin(B * (1 + (1 - 1j) * y) * alpha) + \
           (2 + 2j) * B * y * alpha * np.sinh(B * (1 + (1 + 1j) * y) * alpha)
    return -(1j * expr) / denominator

def T3(y, B, alpha):
    I = 1j
    z  = (0.5 + 0.5*I) * B * alpha
    zy = (0.5 + 0.5*I) * B * (1 + 2*y) * alpha

    term1 = (1 - I) * (B*alpha * np.cos(z) - (1 - I) * np.sin(z))
    term2 = (-1 + I) * np.cosh(zy) + B*y*alpha * np.sinh(zy)

    denom = (B**3) * (alpha**3) * (np.cos(B*alpha) + np.cosh(B*alpha))
    return -(term1 * term2) / denom


def T4(y, B, alpha):
    I = 1j
    z = (0.5 + 0.5*I) * B * alpha
    return y**2 * (-1 + ((1 - I) * np.tan(z)) / (B * alpha))


def T5(y, B, alpha):
    I = 1j
    z  = (0.5 + 0.5*I) * B * alpha
    zy = (0.5 + 0.5*I) * B * (1 + 2*y) * alpha
    termA = B**2 * y**2 * alpha**2 + I * np.cosh(zy) / np.cosh(z)
    termB = B * alpha - (1 - I) * np.tan(z)
    return (termA * termB) / (B**3 * alpha**3)


def T0(B, alpha):
    I = 1j
    num = (0.75 + 0.75*I) * (I * np.cos(B * alpha) + np.cosh(B * alpha))
    denom = B**2 * alpha**2 * (np.cos(B * alpha) + np.cosh(B * alpha))
    return num / denom

def Tm(B, alpha):
    I = 1j
    term1 = 0.5
    term2 = (0.75 + 0.75*I) / (B**2 * alpha**2)
    term3 = (3 * np.cos(B * alpha)) / (2 * B**2 * alpha**2 * (np.cos(B * alpha) + np.cosh(B * alpha)))
    return term1 + term2 - term3

def tau(B, alpha):
    """
    T(-1) - T0
    """
    return Tm(B, alpha) - T0(B, alpha)




def B1(B, alpha):
    I = 1j
    num = (
        (12*I - (9 - 9*I) * B**2 * alpha**2 + B**4 * alpha**4) * np.cos(B * alpha)
        + (-12*I + (9 + 9*I) * B**2 * alpha**2 + B**4 * alpha**4) * np.cosh(B * alpha)
        + (33 - 3*I) * B * alpha * np.sin(B * alpha)
        - (33 + 3*I) * B * alpha * np.sinh(B * alpha)
    )
    denom = 12 * B**4 * alpha**4 * (np.cos(B * alpha) + np.cosh(B * alpha))
    return -num / denom

def B2(B, alpha):
    z = (0.5 + 0.5j) * B * alpha
    return ((-1j*B*alpha + (1+1j)*np.tan(z)) * (B*alpha - (1-1j)*np.tanh(z))) / (B**2 * alpha**2)

def B3(B, alpha):
    I = 1j
    z = (0.5 + 0.5*I) * B * alpha
    return ((-I * B * alpha + (1 + I) * np.tan(z)) * (B * alpha - (1 - I) * np.tanh(z))) / (B**2 * alpha**2)

def Phi(x, B, alpha, forcing_funcs):
    B=1/np.sqrt(2)
    G1 = forcing_funcs["G1"]
    G2= forcing_funcs["G2"]
    Gamma = forcing_funcs["Gamma"]

    return alpha**2 * B1(B, alpha) * Gamma(x) + B2(B, alpha)* G1(x) + B3(B, alpha) * G2(x)

def C2(alpha, A, eps, forcing_funcs):
    B=1/np.sqrt(2)
    coef = (12*kappa(alpha, A, eps))/(alpha**2)

    return -coef*Phi(0, B, alpha, forcing_funcs)

def C1(alpha, A, eps, forcing_funcs):
    B=1/np.sqrt(2)
    coef = (12*kappa(alpha, A, eps))/(alpha**2)

    return coef*(Phi(0, B, alpha, forcing_funcs) - Phi(1, B, alpha, forcing_funcs))


def p1no(x, alpha, A, eps, forcing="sin", k=1):
    if forcing == "sin":
        forcing_funcs = load_default_forcing(k)
    elif forcing == "uniform":
        forcing_funcs = uniform_forcing()
    elif forcing == "cos":
        forcing_funcs = cos_forcing(k)
    else:
        forcing_funcs = forcing

    B = 1 / np.sqrt(2)
    coef = (12 * kappa(alpha, A, eps)) / (alpha**2)

    term1 = coef * Phi(x, B, alpha, forcing_funcs)
    term2 = C1(alpha, A, eps, forcing_funcs) * x
    term3 = C2(alpha, A, eps, forcing_funcs)

    return 2 * np.real(term1 + term2 + term3)

def p1nox(x, alpha, A, eps, forcing="sin", k=1):
    if forcing == "sin":
        forcing_funcs = load_default_forcing(k)
    elif forcing == "uniform":
        forcing_funcs = uniform_forcing(k)
    else:
        forcing_funcs = forcing

    B=1/np.sqrt(2)
    g1 = forcing_funcs["g1"]
    g2= forcing_funcs["g2"]
    gamma = forcing_funcs["gamma"]
    coef = (12*kappa(alpha, A, eps))/(alpha**2)

    term1 = coef*( alpha**2 * gamma(x)*B1(B, alpha) + g2(x)*B3(B, alpha) + g1(x)*B2(B, alpha))
    term2 = C1(alpha, A, eps, forcing_funcs)
    return 2*np.real(term1 + term2)

def px(x, alpha, A, eps, forcing_funcs):
    B=1/np.sqrt(2)
    g1 = forcing_funcs["g1"]
    g2= forcing_funcs["g2"]
    gamma = forcing_funcs["gamma"]
    coef = (12*kappa(alpha, A, eps))/(alpha**2)

    term1 = coef*(alpha**2 * gamma(x)*B1(B, alpha) + g2(x)*B3(B, alpha) + g1(x)*B2(B, alpha))
    term2 = C1(alpha, A, eps, forcing_funcs)
    return term1 + term2

def pxx(x, alpha, A, eps, forcing_funcs):
    B=1/np.sqrt(2)
    g1x = forcing_funcs["g1_x"]
    g2x= forcing_funcs["g2_x"]
    gammax = forcing_funcs["gamma_x"]
    coef = (12*kappa(alpha, A, eps))/(alpha**2)

    term1 = coef*( alpha**2 * gammax(x)*B1(B, alpha) + g2x(x)*B3(B, alpha) + g1x(x)*B2(B, alpha))
    return term1

def u(x, y, alpha, A, eps, forcing="sin", k=1):
    if forcing == "sin":
        forcing_funcs = load_default_forcing(k)
    elif forcing == "uniform":
        forcing_funcs = uniform_forcing()
    elif forcing == 'cos':
        forcing_funcs = cos_forcing(k)
    else:
        forcing_funcs = forcing
    B = 1 / np.sqrt(2)
    
    gamma = forcing_funcs["gamma"]
    
    u_val = px(x, alpha, A, eps, forcing_funcs) * (y**2 + y)/2 + kappa(alpha, A, eps) * gamma(x) *(T1(y, B, alpha)+T2(y, B, alpha)+ T3(y, B, alpha) + T4(y, B, alpha) + T5(y, B, alpha) + tau(B, alpha)*y -T0(B, alpha))

    return u_val

def R1(y, B, alpha):
    denominator = 24 * B**3 * alpha**3 * (np.cos(B * alpha) + np.cosh(B * alpha))
    
    term1 = 4 * B**3 * y**3 * alpha**3 * np.cos(B * alpha)
    term2 = 4 * B**3 * y**3 * alpha**3 * np.cosh(B * alpha)
    term3 = 15 * np.sin(B * alpha)
    term4 = -12 * np.cosh(B * (1 + y) * alpha) * np.sin(B * y * alpha)
    term5 = -12 * np.cosh(B * y * alpha) * np.sin(B * (1 + y) * alpha)
    term6 = -3 * np.sin(B * (1 + 2 * y) * alpha)
    term7 = -15 * np.sinh(B * alpha)
    term8 = 12 * np.cos(B * (1 + y) * alpha) * np.sinh(B * y * alpha)
    term9 = 12 * np.cos(B * y * alpha) * np.sinh(B * (1 + y) * alpha)
    term10 = 3 * np.sinh(B * (1 + 2 * y) * alpha)

    numerator = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10

    return -numerator / denominator

def R2(y, B, alpha):
    denominator = 8 * B**3 * alpha**3 * (np.cos(B * alpha) + np.cosh(B * alpha))
    
    term1 = 4 * B * y * alpha * np.cos(B * (1 + (1 - 1j) * y) * alpha)
    term2 = 4 * B * y * alpha * np.cosh(B * (1 + (1 + 1j) * y) * alpha)
    term3 = (3 + 8j) * np.sin(B * alpha)
    term4 = -(4 + 8j) * np.sin(B * (1 + (1 - 1j) * y) * alpha)
    term5 = np.sin(B * (1 + 2 * y) * alpha)
    term6 = (3 - 8j) * np.sinh(B * alpha)
    term7 = -(4 - 8j) * np.sinh(B * (1 + (1 + 1j) * y) * alpha)
    term8 = np.sinh(B * (1 + 2 * y) * alpha)
    
    numerator = 1j * (term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8)
    
    return numerator / denominator

def R3(y, B, alpha):
    I = 1j
    z  = (0.5 + 0.5*I) * B * alpha
    z2 = (1 + I) * B * (0.5 + y) * alpha

    num = (0.5 + 0.5*I) * (B*alpha * np.cos(z) - (1 - I) * np.sin(z)) * (
        (1 + I) * B * y * alpha * np.cosh(z2)
        + 3 * np.sinh(z)
        - 3 * np.sinh(z2)
    )
    denom = B**4 * alpha**4 * (np.cos(B * alpha) + np.cosh(B * alpha))
    return -num / denom



def R4(y, B, alpha):
    I = 1j
    z = (0.5 + 0.5*I) * B * alpha
    return (y**3 * (B * alpha - (1 - I) * np.tan(z))) / (3 * B * alpha)

def R41(y, B, alpha):
    #Integral of T5
    I = 1j
    z  = (0.5 + 0.5*I) * B * alpha
    z2 = (1 + I) * B * (0.5 + y) * alpha

    num = (1/3 + (1/3)*I) * (B*alpha * np.cos(z) - (1 - I) * np.sin(z)) * (
        (-1 + I) * B**3 * y**3 * alpha**3 * np.cosh(z)
        + 3 * np.sinh(z)
        - 3 * np.sinh(z2)
    )
    denom = B**4 * alpha**4 * (np.cos(B * alpha) + np.cosh(B * alpha))
    return num / denom

def R5(y):
    return -(y**2) / 4

def R6(y, B, alpha):
    I = 1j
    num = (0.75 + 0.75*I) * y * (I * np.cos(B * alpha) + np.cosh(B * alpha))
    denom = B**2 * alpha**2 * (np.cos(B * alpha) + np.cosh(B * alpha))
    return num / denom

def R7(y, B, alpha):
    y = np.asarray(y)
    z   = (0.5 + 0.5j) * B * alpha
    z_y = (0.5 + 0.5j) * B * (1 + 2*y) * alpha
    pref = (1+1j) * (B*alpha*np.cos(z) - (1-1j)*np.sin(z))
    inner = (1+1j)*B*y*alpha*np.cosh(z_y) + np.sinh(z) - np.sinh(z_y)
    denom = B**2 * alpha**2 * (np.cos(B*alpha) + np.cosh(B*alpha))
    return pref * inner / denom

def R8(y, B, alpha):
    y = np.asarray(y)
    z   = (0.5 + 0.5j) * B * alpha
    z_y = (0.5 + 0.5j) * B * (1 + 2*y) * alpha
    pref = (1+1j) * (B*alpha*np.cos(z) - (1-1j)*np.sin(z))
    inner = (1+1j)*B*y*alpha*np.cosh(z) + np.sinh(z) - np.sinh(z_y)
    denom = B**2 * alpha**2 * (np.cos(B*alpha) + np.cosh(B*alpha))
    return pref * inner / denom

def v(x, y, alpha, A, eps, forcing="sin", k=1):
    if forcing == "sin":
        forcing_funcs = load_default_forcing(k)
    elif forcing == "uniform":
        forcing_funcs = uniform_forcing()
    elif forcing == 'cos':
        forcing_funcs = cos_forcing(k)
    else:
        forcing_funcs = forcing
    B=1/np.sqrt(2)

    g1x = forcing_funcs["g1_x"]
    g2x= forcing_funcs["g2_x"]
    gammax = forcing_funcs["gamma_x"]

    R = R1(y, B, alpha)+R2(y, B, alpha)+R3(y, B, alpha) +R4(y, B, alpha) + R41(y, B, alpha)+R5(y)+R6(y, B, alpha)
    term1 = -(pxx(x, alpha, A, eps, forcing_funcs) /2) * (y**3/3 + y**2/2)
    term2 = kappa(alpha, A, eps)* gammax(x) * R
    term3 = (kappa(alpha, A, eps)/(alpha**2)) * (g1x(x)* R7(y, B, alpha) + g2x(x)*R8(y, B, alpha))
    # Assemble v(x, y)
    v_val = term1 + term2 + term3

    return v_val

def u1no(x, y, alpha, A, eps, forcing="sin", k=1):
    return 2*np.real(u(x, y, alpha, A, eps, forcing, k))

def v1no(x, y, alpha, A, eps, forcing="sin", k=1):
    return 2*np.real(v(x, y, alpha, A, eps, forcing, k))