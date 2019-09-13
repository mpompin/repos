import numpy as np
from causalhenonmaps import causalhenonmaps
from PMIMEsig import PMIMEsig

K = 5
C = 0.2
n = 500
nnei = 5
Lmax = 5
thres = 0.03
T = 1
nsur = 100
alpha = 0.05
showtxt = 2

xM = causalhenonmaps(K, C, n)
RM, ecC = PMIMEsig(xM, Lmax, T, nnei, nsur, alpha, showtxt)
print(RM)
