import numpy as np

def B11(B, alpha):
    complex_term1 = (-12j + (9 + 21j) * B**2 * alpha**2 - B**4 * alpha**4) * np.cos(B * alpha)
    complex_term2 = (12j - (9 - 21j) * B**2 * alpha**2 - B**4 * alpha**4) * np.cosh(B * alpha)
    complex_term3 = (3 + 3j) * B * alpha * (((-10 + 1j) + (1 + 1j) * B**2 * alpha**2) * np.sin(B * alpha) - ((-1 + 10j) + (1 + 1j) * B**2 * alpha**2) * np.sinh(B * alpha))
    
    denominator = 12 * B**4 * alpha**4 * (np.cos(B * alpha) + np.cosh(B * alpha))
    
    result = (complex_term1 + complex_term2 + complex_term3) / denominator
    
    return result

def new_B11(B, alpha):
    I = 1j
    Ba = B * alpha

    term1 = -3 * (4*I - (3 + 7*I) * (B**2) * (alpha**2) + (B**4) * (alpha**4)) * np.cos(Ba)
    term2 = -3 * (-4*I + (3 - 7*I) * (B**2) * (alpha**2) + (B**4) * (alpha**4)) * np.cosh(Ba)

    inner1 = ((-30 + 3*I) + (3 + 1*I) * (B**2) * (alpha**2)) * np.sin(Ba)
    inner2 = ((-3 + 30*I) + (1 + 3*I) * (B**2) * (alpha**2)) * np.sinh(Ba)
    term3 = (1 + I) * B * alpha * (inner1 - inner2)

    numerator = term1 + term2 + term3
    denominator = 12 * (B**4) * (alpha**4) * (np.cos(Ba) + np.cosh(Ba))

    return numerator / denominator

def sech(x):
    return 1 / np.cosh(x)

def T1(B, alpha, y):
    complex_term = (1/2 + 1j/2)
    term1 = B**2 * y**2 * alpha**2
    term2 = 1j * np.cosh(complex_term * B * (1 + 2 * y) * alpha) * sech(complex_term * B * alpha)
    term3 = B * alpha - (1 - 1j) * np.tan(complex_term * B * alpha)
    
    numerator = (term1 + term2) * term3
    denominator = B**3 * alpha**3
    
    result = numerator / denominator
    
    return result
def T2(B, alpha, y):
    term1 = 2 * B**2 * y**2 * alpha**2 * np.cos(B * alpha)
    term2 = 2 * B**2 * y**2 * alpha**2 * np.cosh(B * alpha)
    term3 = 2j * (np.cos(B * (1 + (1 - 1j) * y) * alpha) - np.cos(B * (1 + (1 + 1j) * y) * alpha))
    term4 = -np.cos(B * (1 + 2 * y) * alpha)
    term5 = -2j * np.cosh(B * (1 + (1 - 1j) * y) * alpha)
    term6 = 2j * np.cosh(B * (1 + (1 + 1j) * y) * alpha)
    term7 = np.cosh(B * (1 + 2 * y) * alpha)
    
    numerator = term1 + term2 + term3 + term4 + term5 + term6 + term7
    denominator = 4 * B**2 * alpha**2 * (np.cos(B * alpha) + np.cosh(B * alpha))
    
    result = numerator / denominator
    
    return result
def T3(B, alpha, y):
    term1 = (-2 - 4j) * np.cos(B * (1 + (1 + 1j) * y) * alpha)
    term2 = 1j * np.cos(B * (1 + 2 * y) * alpha)
    term3 = (2 - 4j) * np.cosh(B * (1 + (1 - 1j) * y) * alpha)
    term4 = 1j * np.cosh(B * (1 + 2 * y) * alpha)
    term5 = (2 - 2j) * B * y * alpha * np.sin(B * (1 + (1 + 1j) * y) * alpha)
    term6 = (2 + 2j) * B * y * alpha * np.sinh(B * (1 + (1 - 1j) * y) * alpha)
    
    numerator = term1 + term2 + term3 + term4 + term5 + term6
    denominator = 4 * B**2 * alpha**2 * (np.cos(B * alpha) + np.cosh(B * alpha))
    
    result = numerator / denominator
    
    return result

def T4(B, alpha, y):
    complex_prefactor = (1/2 - 1j/2)
    exp1 = np.exp((-1 - 1j) * B * y * alpha)
    part1 = -1 + (1 + 1j) / (B * alpha) - (2 + 2j) / (B * alpha + B * np.exp((1 - 1j) * B * alpha) * alpha)
    exp2 = np.exp((1 + 1j) * B * (1 + 2 * y) * alpha)
    part2 = ((-1 + 1j) - B * y * alpha + exp2 * ((-1 + 1j) + B * y * alpha))
    
    numerator = complex_prefactor * exp1 * part1 * part2
    denominator = B**2 * (1 + np.exp((1 + 1j) * B * alpha)) * alpha**2
    
    result = numerator / denominator
    
    return result

def newT(B, alpha, y):

    return T1(B, alpha, y) + T2(B, alpha, y) + T3(B, alpha, y) + T4(B, alpha, y)

def B1(B, alpha):
    # Common expressions
    B_alpha = B * alpha
    cos_B_alpha = np.cos(B_alpha)
    cosh_B_alpha = np.cosh(B_alpha)
    sin_B_alpha = np.sin(B_alpha)
    sinh_B_alpha = np.sinh(B_alpha)
    
    # Compute each term in the numerator and denominator
    numerator = 3 * B_alpha * cos_B_alpha + 3 * B_alpha * cosh_B_alpha - 2 * sin_B_alpha - 2 * sinh_B_alpha
    denominator = 2 * B_alpha * cos_B_alpha + 2 * B_alpha * cosh_B_alpha
    
    # Calculate the result
    result = numerator / denominator
    
    return result

def T0(B, alpha):
    i = 1j  # The imaginary unit
    
    # Compute Cos[B * alpha] and Cosh[B * alpha]
    cos_B_alpha = np.cos(B * alpha)
    cosh_B_alpha = np.cosh(B * alpha)
    
    # Compute the numerator
    numerator = (3/4 + (3 * i) / 4) * (cos_B_alpha + i * cosh_B_alpha)
    
    # Compute the denominator
    denominator = B**2 * alpha**2 * (cos_B_alpha + cosh_B_alpha)
    
    # Calculate the result
    result = - numerator / denominator
    
    return result

def Tm1(B, alpha):
    i = 1j  # The imaginary unit
    
    # Common expressions
    B_alpha = B * alpha
    cos_B_alpha = np.cos(B_alpha)
    cosh_B_alpha = np.cosh(B_alpha)
    sin_B_alpha = np.sin(B_alpha)
    sinh_B_alpha = np.sinh(B_alpha)
    
    # Decompose the expression into smaller parts
    cos_part = (3 + 3*i) * (-1 + (1 - i) * B**2 * alpha**2) * cos_B_alpha
    cosh_part = ((3 - 3*i) + 6 * B**2 * alpha**2) * cosh_B_alpha
    sin_sinh_part = -4 * B_alpha * (sin_B_alpha + sinh_B_alpha)
    
    # Combine the parts
    numerator = cos_part + cosh_part + sin_sinh_part
    denominator = 4 * B**2 * alpha**2 * (cos_B_alpha + cosh_B_alpha)
    
    # Calculate the result
    result = numerator / denominator
    
    return result


def u1(x ,y ,alpha ,A ,eps):

    beta = alpha * (1 + 1j ) / np.sqrt(2)
    B = 1/ np.sqrt(2)
    xi = (np.exp(beta) - 1) / (np.exp(beta) - np.exp(-beta))
    J = (1/ beta) *( (1 - xi) * (np.exp(beta) - 1) - xi * (np.exp(-beta) - 1) - beta)
    Jc = np.conjugate(J)
    v_tild = -A/(2 * eps)
    
    k = alpha**2 * (1/ (4 * J * np.conjugate(J))) * (2*x - 1) * (1 / 2) #*(A/eps)
    k_prime = alpha**2 * (1/ (4 * J * np.conjugate(J))) 
    
    #u0 =  k * (newT(B, alpha, y) + (Jc) * (y**2 + y) + 6 * B11(B, alpha) * (y**2  + y) + B1(B, alpha) * y - T0(B, alpha)) #-old ss
    u0 =  k * (newT(B, alpha, y)  + 6 * new_B11(B, alpha) * (y**2  + y) + B1(B, alpha) * y - T0(B, alpha)) -(6/4)* 1j * (y**2 + y)  * (2*x - 1) * (1 / 2)
    u = (A/eps) * 2 * np.real(u0 )
    return u

def R1(B, alpha, y):
    complex_prefactor = (1/6 + 1j/6)
    tan_component = (1/2 + 1j/2) * B * alpha
    term1 = B * alpha - (1 - 1j) * np.tan(tan_component)
    term2 = (-1 + 1j) * B**3 * y**3 * alpha**3
    term3 = -3 * sech(tan_component) * np.sinh(tan_component * (1 + 2 * y))
    term4 = 3 * np.tanh(tan_component)
    
    numerator = complex_prefactor * term1 * (term2 + term3 + term4)
    denominator = B**4 * alpha**4
    
    result = numerator / denominator
    
    return result
def R2(B, alpha, y):
    term1 = 4 * B**3 * y**3 * alpha**3 * np.cos(B * alpha)
    term2 = 4 * B**3 * y**3 * alpha**3 * np.cosh(B * alpha)
    term3 = 12 * np.sin(B * alpha)
    term4 = -6 * np.cos(B * (1 + y) * alpha) * np.sin(B * y * alpha)
    term5 = -(6 - 6j) * np.sin(B * (1 + (1 - 1j) * y) * alpha)
    term6 = -(6 + 6j) * np.sin(B * (1 + (1 + 1j) * y) * alpha)
    term7 = -15 * np.sinh(B * alpha)
    term8 = (6 - 6j) * np.sinh(B * (1 + (1 - 1j) * y) * alpha)
    term9 = (6 + 6j) * np.sinh(B * (1 + (1 + 1j) * y) * alpha)
    term10 = 3 * np.sinh(B * (1 + 2 * y) * alpha)
    
    numerator = term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8 + term9 + term10
    denominator = 24 * B**3 * alpha**3 * (np.cos(B * alpha) + np.cosh(B * alpha))
    
    result = -numerator / denominator
    
    return result
def R3(B, alpha, y):
    term1 = 4 * B * y * alpha * np.cos(B * (1 + (1 + 1j) * y) * alpha)
    term2 = 4 * B * y * alpha * np.cosh(B * (1 + (1 - 1j) * y) * alpha)
    term3 = (3 - 8j) * np.sin(B * alpha)
    term4 = -(4 - 8j) * np.sin(B * (1 + (1 + 1j) * y) * alpha)
    term5 = np.sin(B * (1 + 2 * y) * alpha)
    term6 = (3 + 8j) * np.sinh(B * alpha)
    term7 = -(4 + 8j) * np.sinh(B * (1 + (1 - 1j) * y) * alpha)
    term8 = np.sinh(B * (1 + 2 * y) * alpha)
    
    numerator = 1j * (term1 + term2 + term3 + term4 + term5 + term6 + term7 + term8)
    denominator = 8 * B**3 * alpha**3 * (np.cos(B * alpha) + np.cosh(B * alpha))
    
    result = -numerator / denominator
    
    return result

def R4(B, alpha, y):
    complex_prefactor = (1/4 + 1j/4)
    exp_factor1 = np.exp((-1 - 1j) * B * y * alpha)
    exp_factor2 = np.exp((1 - 1j) * B * alpha)
    exp_factor3 = np.exp((1 + 1j) * B * y * alpha)
    exp_factor4 = np.exp((1 + 1j) * B * alpha)
    exp_factor5 = np.exp((1 + 1j) * B * (1 + 2 * y) * alpha)
    
    part1 = -1 + (1 + 1j) / (B * alpha) - (2 + 2j) / (B * alpha + B * exp_factor2 * alpha)
    part2 = 3 + 3 * exp_factor3 * (-1 + exp_factor4) + (1 + 1j) * B * y * alpha + exp_factor5 * (-3 + (1 + 1j) * B * y * alpha)
    
    numerator = complex_prefactor * exp_factor1 * part1 * part2
    denominator = B**3 * (1 + exp_factor4) * alpha**3
    
    result = numerator / denominator
    
    return result
def newR(B, alpha, y):
    
    return R1(B, alpha, y) + R2(B, alpha, y) + R3(B, alpha, y) + R4(B, alpha, y)
 
def v1(x ,y, alpha ,A ,eps):
    beta = alpha * (1 + 1j ) / np.sqrt(2)
    B = 1/ np.sqrt(2)
    xi = (np.exp(beta) - 1) / (np.exp(beta) - np.exp(-beta))
    J = (1/ beta) *( (1 - xi) * (np.exp(beta) - 1) - xi * (np.exp(-beta) - 1) - beta)
    F = (1/beta) * ((1 - xi) * (np.exp(-beta * y) - 1) - xi * (np.exp(beta * y) - 1) + beta * y)
    v_tild = A/(2 * eps)
    
    #k = alpha**2 * v_tild**2 * (1/ (J * np.conjugate(J))) * (2*x - 1) * (1 / 2) * (A/eps)
    k_prime = alpha**2 * (1/4) * (1/ (J * np.conjugate(J))) 

    #v1 = k_prime * (newR(B, alpha, y) -  (np.conjugate(J) + 6 * B11(alpha, B)) * (y**3 / 3 + y**2 /2)  - B1(alpha, B) *(y**2 /2)  + T0(alpha, B) * y) - ((1j * F)/(4*J))
    v1 = k_prime * (newR(B, alpha, y) -  ((6/4)*(-1j * k_prime) + 6 * new_B11(alpha, B)) * (y**3 / 3 + y**2 /2)  - B1(alpha, B) *(y**2 /2)  + T0(alpha, B) * y) - ((1j * F)/(4*J))

    return (A/eps) * 2*np.real(v1) 