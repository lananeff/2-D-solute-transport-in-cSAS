from .non_uniform_steady_streaming_velocity import u1no, v1no
from .non_uniform_stokes_drift_velocity import usno, vsno

# A/eps factor updated from new scaling
def uLno(x, y, alpha, A, eps, forcing, k=1):
    return u1no(x, y, alpha, A, eps, forcing, k) + usno(x, y, alpha, A, eps, forcing, k)

def vLno(x, y, alpha, A, eps, forcing, k=1):
    return v1no(x, y, alpha, A, eps, forcing, k) +  vsno(x, y, alpha, A, eps, forcing, k)