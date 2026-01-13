
from .steady_streaming_velocity import u1, v1
from .stokes_drift_velocity import us, vs

# A/eps factor updated from new scaling
def uL(x, y, alpha, A, eps):
    return u1(x, y, alpha, A, eps) + us(x, y, alpha, A, eps)

def vL(x, y, alpha, A, eps):
    return v1(x, y, alpha, A, eps) +  vs(x, y, alpha, A, eps)