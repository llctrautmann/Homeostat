import numpy as np

def near(a, b, tol=1E-6):
    return np.fabs(a - b) < tol
