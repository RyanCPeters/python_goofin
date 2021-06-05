import numpy as np
from numba.types import float64
from src import *

TPB_LONG = 16
TPB_MED = 4
TPB_SHORT = 1
GLOBAL_DTYPE = np.float64
GLOBAL_DTYPE_BYTES = GLOBAL_DTYPE(0).nbytes
GLOBAL_NUMBA_DTYPE = float64
GLOBAL_SPB = int(2**7)