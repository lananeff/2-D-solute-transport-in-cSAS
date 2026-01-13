import numpy as np
import math

# Parameters for cardiac, respiratory and slow wave oscillations in mice and humans
# Real value of D
amyloid_beta_D = 1.8e-10
nu_real = 8e-7 #m^2/s nu=mu/rho

real_Schmidt =  nu_real/amyloid_beta_D
small_ion_Schmidt = 267
# SAS thickness
human_h_min_real= 1.45e-3 -0.61e-3
human_h_max_real= 1.45e-3 +0.61e-3
human_h_real = 1.45e-3
mice_h_min_real=3e-5
mice_h_max_real=5e-5
mice_h_real=4e-5

# Brain radius
human_r_real = 0.067
mice_r_real = 4.63e-3

# Amplitude oscillation
human_ca_min_real=3.7e-6
human_ca_real = 6.6e-6
human_ca_max_real=9.5e-6
# human_ra_min_real= human_ca_min_real/3
# human_ra_max_real= human_ca_max_real/3
human_ra_min_real= (3.45-1.525)*10**(-6)
human_ra_max_real= (3.45+1.525)*10**(-6)
human_sa_min_real= (49.5-21.75)*10**(-6)
human_sa_max_real= (49.5+21.75)*10**(-6)

# 
mice_ca_real= 3.06e-6
mice_ca_max_real= 3.06e-6 + 0.333e-6
mice_ca_min_real= 3.06e-6 -0.333e-6
#mice_ra_min= 
#mice_ra_max=

mice_sa_max_real = 14.6065e-6
mice_sa_min_real = 0.3046e-6
mice_sa_real = (mice_sa_max_real + mice_sa_min_real)/2



# Frequency of oscillation
human_comega_real= ((1+1.6)/2) * 2 * np.pi
human_comega_min_real = 1 * 2 * np.pi
human_comega_max_real = 1.6 * 2 * np.pi
human_romega_min_real= 0.08 * 2 * np.pi
human_romega_max_real= 0.3 * 2 * np.pi
human_romega_real = ((0.08+0.3)/2) *2 * np.pi
human_somega_min_real= 0.05 * 2 * np.pi
human_somega_max_real= 0.1 * 2 * np.pi
human_somega_real = ((0.05+0.1)/2) *2 * np.pi

mice_comega_real= 10 * 2 * np.pi
mice_comega_min_real= 8 * 2 * np.pi
mice_comega_max_real= 12 * 2 * np.pi
mice_romega_min_real= 4.25 * 2 * np.pi
mice_romega_max_real= 4.25 * 2 * np.pi
mice_romega_real = (4.25) *2 * np.pi
mice_somega_min_real= 0.1 * 2 * np.pi
mice_somega_max_real= 0.3 * 2 * np.pi
mice_somega_real = ((0.1+0.3)/2) *2 * np.pi


# Scaled amplitudes
human_ca_min= human_ca_min_real/human_h_max_real
human_ca_max= human_ca_max_real/human_h_min_real
human_ra_min= human_ra_min_real/human_h_max_real
human_ra_max= human_ra_max_real/human_h_min_real
human_sa_min= human_sa_min_real/human_h_max_real
human_sa_max= human_sa_max_real/human_h_min_real

mice_ca_min= mice_ca_min_real/mice_h_max_real
mice_ca_max= mice_ca_max_real/mice_h_min_real
mice_sa_min= mice_sa_min_real/mice_h_max_real
mice_sa_max= mice_sa_max_real/mice_h_min_real

#  Womersley numbers alpha^2!
human_c_alpha_min= 1/nu_real * human_h_min_real**2 * human_comega_min_real
human_c_alpha_max= 1/nu_real * human_h_max_real**2 * human_comega_max_real
human_c_alpha= 1/nu_real * (human_h_real**2) * human_comega_real
human_r_alpha_min= 1/nu_real * human_h_min_real**2 * human_romega_min_real
human_r_alpha_max= 1/nu_real * human_h_max_real**2 * human_romega_max_real
human_r_alpha = 1/nu_real * (human_h_real**2) * human_romega_real
human_s_alpha_min= 1/nu_real * human_h_min_real**2 * human_somega_min_real
human_s_alpha_max= 1/nu_real * human_h_max_real**2 * human_somega_max_real
human_s_alpha = 1/nu_real * (human_h_real**2) * human_somega_real

mice_c_alpha_min = 1/nu_real * mice_h_min_real**2 * mice_comega_min_real
mice_c_alpha_max = 1/nu_real * mice_h_max_real**2 * mice_comega_max_real
mice_c_alpha     = 1/nu_real * (mice_h_real**2) * mice_comega_real

mice_r_alpha_min = 1/nu_real * mice_h_min_real**2 * mice_romega_min_real
mice_r_alpha_max = 1/nu_real * mice_h_max_real**2 * mice_romega_max_real
mice_r_alpha     = 1/nu_real * (mice_h_real**2) * mice_romega_real

mice_s_alpha_min = 1/nu_real * mice_h_min_real**2 * mice_somega_min_real
mice_s_alpha_max = 1/nu_real * mice_h_max_real**2 * mice_somega_max_real
mice_s_alpha     = 1/nu_real * (mice_h_real**2) * mice_somega_real

# Epsilon aspect rations
human_eps_min= human_h_min_real/(human_r_real*2)
human_eps_max = human_h_max_real/ (human_r_real*2)
mice_eps_min = mice_h_min_real/(mice_r_real*2)
mice_eps_max = mice_h_max_real/(mice_r_real*2)

# Production flow - rate/ surface area
human_prod_min = 5e-9/ (np.pi *2 * human_r_real * human_h_max_real)
human_prod_max = 2e-8/ (np.pi *2 * human_r_real * human_h_min_real)
mice_prod_min = 6.6e-13
mice_prod_max = 5.8e-12

# Parameters chosen for round peosc numbers
S_pe = 4096
A_pe = 0.0025
alpha_pe = 2.5
omega_pe = (alpha_pe**2 * nu_real)/human_h_real**2

PARAMETERS = {
    "D_amyloid" : amyloid_beta_D,
    
    
    "human" : {
        "nu" : nu_real, 
        "h_min" : human_h_min_real,
        "h_max" : human_h_max_real,
        "h" : (human_h_min_real + human_h_max_real) /2,
        "r" : human_r_real,
        "L" : human_r_real *2,
        "eps_min" : human_eps_min,
        "eps_max" : human_eps_max,
        "eps" : (human_eps_max + human_eps_min) /2,
        "S": S_pe,
        "A": A_pe,
        "alpha": alpha_pe,
        "omega": omega_pe,
        "cardiac" : {
            "omega" : human_comega_real,
            "omega_min": human_comega_min_real,
            "omega_max": human_comega_max_real,
            "A_min" : human_ca_min,
            "A_max" : human_ca_max,
            "A" : ((human_ca_min_real + human_ca_max_real) /2) / ((human_h_min_real + human_h_max_real)/2),
            "alpha_min" : math.sqrt(human_c_alpha_min),
            "alpha_max" : math.sqrt(human_c_alpha_max),
            "alpha" : np.sqrt(human_c_alpha)
            
        },
        "respiratory": {
            "omega_min": human_romega_min_real,
            "omega_max": human_romega_max_real,
            "A_min": human_ra_min,
            "A_max": human_ra_max,
            "A": ((human_ra_min_real + human_ra_max_real) /2) / ((human_h_min_real + human_h_max_real)/2),
            "alpha_min": math.sqrt(human_r_alpha_min),
            "alpha_max": math.sqrt(human_r_alpha_max),
            "alpha": np.sqrt(human_r_alpha),
        },
        "sleep": {
            "omega_min": human_somega_min_real,
            "omega_max": human_somega_max_real,
            "A_min": human_sa_min,
            "A_max": human_sa_max,
            "A": ((human_sa_min_real + human_sa_max_real) /2) / ((human_h_min_real + human_h_max_real)/2),
            "alpha_min": math.sqrt(human_s_alpha_min),
            "alpha_max": math.sqrt(human_s_alpha_max),
            "alpha": np.sqrt(human_s_alpha),
        },
    },
    "mouse": {
        "h_min": mice_h_min_real,
        "h_max": mice_h_max_real,
        "h": (mice_h_min_real + mice_h_max_real) / 2,
        "r": mice_r_real,
        "eps_min": mice_eps_min,
        "eps_max": mice_eps_max,
        "eps": (mice_eps_min + mice_eps_max) / 2,
        "S": nu_real/amyloid_beta_D,
        "cardiac": {"omega" : mice_comega_real,
            "A_min" : mice_ca_min,
            "A_max" : mice_ca_max,
            "A" : ((mice_ca_min_real + mice_ca_max_real) /2) / ((mice_h_min_real + mice_h_max_real)/2),
            "alpha_min" : math.sqrt(mice_c_alpha_min),
            "alpha_max" : math.sqrt(mice_c_alpha_max),
            "alpha" : np.sqrt(mice_c_alpha)},
        "respiratory": {},
        "sleep": {"omega_min": mice_somega_min_real,
            "omega_max": mice_somega_max_real,
            "A_min": mice_sa_min,
            "A_max": mice_sa_max,
            "A": ((mice_sa_min_real + mice_sa_max_real) /2) / ((mice_h_min_real + mice_h_max_real)/2),
            "alpha_min": math.sqrt(mice_s_alpha_min),
            "alpha_max": math.sqrt(mice_s_alpha_max),
            "alpha": np.sqrt(mice_s_alpha)},
    },
}