import numpy as np
from scipy.special import psi
from ANN import ANN, nneighforgivenr


def PMIMEsig(allM, Lmax=5, T=1, nnei=5, nsur=100, alpha=0.05, showtxt=1):
    '''
    function [RM,ecC] = PMIMEsig(allM,Lmax,T,nnei,nsur,alpha,showtxt)
    PMIMEsig (Partial Mutual Information on Mixed Embedding, Significance)
    computes the measure R_{X->Y|Z} for all combinations of X and Y time
    series from the multivariate time series given in matrix 'allM', of size
    N x K, where Z contains the rest K-2 time series.
    The components of X,Y, and Z, are found from a mixed embedding aiming at
    explaining Y. The mixed embedding is formed by using the progressive
    embedding algorithm based on conditional mutual information (CMI).
    CMI is estimated by the method of nearest neighbors (Kraskov's method).
    The function is the same as PMIME.m but defines the stopping criterion
    differently, using an adjusted rather than fixed threshold. Specifically,
    a significance test is performed for the CMI of the new component
    selected as best. For the significance test a number 'nsur' surrogates
    are given and the test decision is made at a given significance level
    'alpha'. The ramdomized data are obtained using rand permutation of the
    time indices of the candidate lag variable and indpependent rand
    permutation of the lagged variables already selected.
    We experienced that in rare cases the null hypothesis of the significance
    test is always rejected and the algorithm does not terminate. Therefore
    we included a second condition for termination of the algorithm when the
    ratio I(x^F; w| wemb) / I(x^F; w,wemb) increases in the last two embedding
    cycles, where I(x^F; w| wemb) is the CMI of the selected lagged variable
    w and the future response state x^F given the current mixed embedding
    vector, and I(x^F; w,wemb) is the MI between x^F and the augmented mixed
    embedding vector [wemb w].
    The derived R measure indicates the information flow of time series X to
    time series Y conditioned on the rest time series in Z. The measure
    values are stored in a K x K matrix 'RM' and given to the output, where
    the value at position (i,j) indicates the effect from i to j (row to
    col), and the (i,i) components are zero.
    INPUTS
    - allM : the N x K matrix of the K time series of length N.
    - Lmax : the maximum delay to search for X and Y components for the mixed
             embedding vector [default is 5].
    - T    : T steps ahead that the mixed embedding vector has to explain.
             Note that if T>1 the future vector is of length T and contains
             the samples at times t+1,..,t+T [dafault is 1].
    - nnei : number of nearest neighbors for density estimation [default is 5]
    - nsur : the number of surrogates for the significance test [default is 100]
    - alpha: the significance level for the test for the termination
             criterion [default is 0.05].
    - showtxt : if 0 or negative do not print out anything,
                if 1 print out the response variable index at each run,
                if 2 or larger print also info for each embedding cycle [default is 1].
    OUTPUTS
    - RM   : A K x K matrix containing the R values computed by PMIME using
             the surrogates for setting the stopping criterion.
    - ecC  : cell array of K components, where each component is a matrix of
             size E x 5, and E is the number of embedding cycles. For each
             embedding cycle the following 5 results are stored:
             1. variable index, 2. lag index, 3. CMI of the selected lagged
             variable w and the future response state x^F given the current
             mixed embedding vector, I(x^F; w| wemb). 4. MI between x^F and
             the augmented mixed embedding vector [wemb w], I(x^F; w,wemb).
             5. The ration of 3. and 4.: I(x^F; w| wemb)/I(x^F; w,wemb)

    Copyright (C) 2015 Dimitris Kugiumtzis
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.

    =========================================================================
    Reference : D. Kugiumtzis, "Direct coupling information measure from
                non-uniform embedding", Physical Review E, Vol 87, 062918,
                2013
                I. Vlachos, D. Kugiumtzis, "Non-uniform state space
                reconstruction and coupling detection", Physical Review E,
                Vol 82, 016207, 2010
    Link      : http://users.auth.gr/dkugiu/
    =========================================================================
    '''
    maxcomps = 20  # A safeguard, to make sure that the algorithm does not make more than "maxcomps" embedding cycles.
    N, K = allM.shape #K time series, N length
    wV = np.ones((K), dtype=int) * Lmax #vector with max delay
    # Standardization of the input matrix columnwise in [0,1].
    minallV = np.min(allM, axis=0) #min of each var
    rang = np.kron(1. / np.ptp(allM, axis=0), np.ones((N, 1)))# 1/range--> N x 1--->N x K
    allM = np.multiply(allM - np.kron(minallV, np.ones((N, 1))), rang)# x-min/range --> normalize sto [0,1]
    # from sklearn.preprocessing import MinMaxScaler
    # scaler = MinMaxScaler()
    # allM_new = scaler.transform(allM)
    # Build up the lag matrix from all variables
    alllagM = np.full((N, np.sum(wV)), np.nan)  # lag matrix of all variables - matrix N x K with NaN
    indlagM = np.full((K, 2), np.nan, dtype=int)  # Start and end of columns of each variable in lag matrix
    count = 0
    for iK in range(K): #!!!!!mporei na ginei me shift pio grigora
        indlagM[iK, :] = np.array([count, count + wV[iK] - 1])
        alllagM[:, indlagM[iK, 0]] = allM[:, iK]  # lag=0
        for ilag in range(1, wV[iK]):  # lag=1,...,Lmax-1
            alllagM[ilag:, indlagM[iK, 0] + ilag] = allM[:-ilag, iK]
        count = count + wV[iK]
    alllagM = alllagM[Lmax - 1:-T, :] # kovei ton pinaka gia na min exei ta NaN meta ta shift ! dropna()
    N1, alllags = alllagM.shape
    # Find mixed embedding and R measure for purpose: from (X,Y,Z) -> Y
    RM = np.zeros((K, K))
    ecC = [np.array([])] * K
    psinnei = psi(nnei)  # Computed once here, to be called in several times #digamma(ln(Gamma)) twn kNN(input)ψ(k)
    psiN1 = psi(N1)  # Computed once here, to be called in several times #digamma(ln(Gamma)) twn sample(input)ψ(N)
    for iK in range(K): # gia kathe xronoseira
        if showtxt == 1:
            print('%d..' % iK, end='', flush=True)
        elif showtxt >= 2:
            print('Response variable index=%d.. \n' % iK)
            print('EmbeddingCycle  Variable  Lag  I(x^F;w|wemb)  I(x^F;w,wemb)  I(x^F;w|wemb)/I(x^F;w,wemb) \n')
        Xtemp = np.full((N, T), np.nan) # edw tha mpei to future vector (p.x. gia T=1)
        for iT in range(T):
            Xtemp[:-(iT + 1), iT] = allM[iT + 1:, iK] # na koitaei T mprosta !pali me shift()
        xFM = Xtemp[Lmax - 1:-T, :]  # The future vector of response
        # First embedding cycle: max I(y^T, w), over all candidates w
        miV = np.full((alllags), np.nan) #dianysma MI gia ola ta lag (olwn twn xronoseirwn)
        for i1 in range(alllags): # gia kathe x apostasi toy kNN kai average aytvn
            # Compute the mutual information of future response and each one of
            # the candidate lags using the nearest neighbor estimate
            xnowM = np.concatenate((xFM, alllagM[:, i1].reshape((N1, 1))), axis=1) # check kathe lag me to future dianusma
            _, distsM = ANN(xnowM, nnei + 1,metric='chebyshev')
            maxdistV = distsM[:, -1] # i apostasi tou pio makrinoy geitona (kNN)
            nyFV = nneighforgivenr(xFM, maxdistV - np.ones((N1)) * 10 ** (-10),metric='chebyshev')
            nwcandV = nneighforgivenr(alllagM[:, i1].reshape(N1, 1), maxdistV - np.ones((N1)) * 10 ** (-10),metric='chebyshev')
            psibothM = psi(np.concatenate((nyFV.reshape(N1, 1), nwcandV.reshape(N1, 1)), axis=1))
            miV[i1] = psinnei + psiN1 - np.mean(np.sum(psibothM, axis=1)) #I(X;Y) = ψ(k) + ψ(Ν) - <ψ(Nx + 1) + ψ(Ny + 1)>
        maxmi = np.max(miV)
        iembV = np.array([np.argmax(miV)])
        xembM = alllagM[:, iembV].reshape((N1, 1))
        # The randomization signficance test for the mutual information (MI) of
        # selected lagged variable and the future response state
        misurV = np.full((nsur + 1), np.nan)
        misurV[0] = maxmi
        for isur in range(nsur):
            xsurV = xembM[np.random.permutation(N1)]
            xnowM = np.concatenate((xFM, xsurV.reshape((N1, 1))), axis=1)
            _, distsM = ANN(xnowM, nnei + 1)
            maxdistV = distsM[:, -1]
            nyFV = nneighforgivenr(xFM, maxdistV - np.ones((N1)) * 10 ** (-10))
            nwsurV = nneighforgivenr(xsurV.reshape(N1, 1), maxdistV - np.ones((N1)) * 10 ** (-10))
            psibothM = psi(np.concatenate((nyFV.reshape(N1, 1), nwsurV.reshape(N1, 1)), axis=1))
            misurV[isur + 1] = psinnei + psiN1 - np.mean(np.sum(psibothM, axis=1))
        imisurV = np.argsort(misurV)
        rnkx = np.where(imisurV == 0)[0][0] + 1  # The rank of the original MI
        pval = 1 - (rnkx - 0.326) / (nsur + 1 + 0.348)  # One-sided test (p-value is
        # obtained by the rank ordering applying a suggested correction)
        if pval < alpha:
            # add the selected lag variable in the first embedding cycle and
            # show it
            varind = np.floor(iembV[0] / Lmax).astype(int)  # the variable
            lagind = np.mod(iembV[0] + 1, Lmax).astype(int)
            if lagind == 0:
                lagind = Lmax
            ecC[iK] = np.array([varind, lagind, miV[iembV], np.nan, np.nan]).reshape((1, 5))  # For the first component
            if showtxt >= 2:
                print(
                    '%d \t %d \t %d \t %2.5f \t %2.5f \t %2.5f \n' % (ecC[iK].shape[0], ecC[iK][-1, 0], ecC[iK][-1, 1],
                                                                      ecC[iK][-1, 2], ecC[iK][-1, 3], ecC[iK][-1, 4]))
            # End of first embedding cycle, the first lagged variale is found
            terminator = 0  # Flag for terminating the embedding cycles
            maxcomps = min(alllagM.shape[1], maxcomps)  # To avoid large embedding
            # Run iteratively, for each embedding cycle select w from max I(y^; w | wemb)
            while (terminator == 0) & (xembM.shape[1] < maxcomps):
                activeV = np.setdiff1d(np.arange(alllags), iembV)  # The indexed of the candidates
                cmiV = np.full((alllags), np.nan)  # I(y^; w | wemb)
                miwV = np.full((alllags), np.nan)  # I(y^; w, wemb)
                for i1 in activeV:
                    # For each candidate lag w compute I(y^; w | wemb) and I(y^; w, wemb)
                    xallnowM = np.concatenate((xFM, alllagM[:, i1].reshape((N1, 1)), xembM), axis=1)
                    _, distsM = ANN(xallnowM, nnei + 1)
                    maxdistV = distsM[:, -1]
                    nwV = nneighforgivenr(xembM, maxdistV - np.ones((N1)) * 10 ** (-10))
                    nwcandV = nneighforgivenr(np.concatenate((alllagM[:, i1].reshape(N1, 1), xembM), axis=1),
                                              maxdistV - np.ones((N1)) * 10 ** (-10))
                    nyFwV = nneighforgivenr(np.concatenate((xFM, xembM), axis=1),
                                            maxdistV - np.ones((N1)) * 10 ** (-10))
                    psinowM = np.full((N1, 3), np.nan)
                    psinowM[:, 0] = psi(nyFwV)
                    psinowM[:, 1] = psi(nwcandV)
                    psinowM[:, 2] = -psi(nwV)
                    cmiV[i1] = psinnei - np.mean(np.sum(psinowM, axis=1))
                    nyFV = nneighforgivenr(xFM, maxdistV - np.ones((N1)) * 10 ** (-10))
                    psinowM = np.concatenate((psi(nyFV).reshape((N1, 1)), psinowM[:, 1].reshape((N1, 1))), axis=1)
                    miwV[i1] = psinnei + psiN1 - np.mean(np.sum(psinowM, axis=1))
                maxcmi = np.nanmax(cmiV)
                ind = np.nanargmax(cmiV)  # ind: index of the selected lagged variable
                xVnext = alllagM[:, ind]
                varind = np.floor(ind / Lmax)  # the variable
                lagind = np.mod(ind + 1, Lmax)  # the lag of the variable
                if lagind == 0:
                    lagind = Lmax
                # The termination criterion of the surrogate significance test
                cmisurV = np.full((nsur + 1), np.nan)
                cmisurV[0] = maxcmi
                for isur in range(nsur):
                    indxsurV = np.random.permutation(N1)
                    xVnextsur = xVnext[indxsurV]
                    indxsur2V = np.random.permutation(N1)
                    xembMsur = xembM[indxsur2V, :]
                    xallnowM = np.concatenate((xFM, xVnextsur.reshape((N1, 1)), xembMsur), axis=1)
                    _, distsM = ANN(xallnowM, nnei + 1)
                    maxdistV = distsM[:, -1]
                    nwsurV = nneighforgivenr(xembMsur, maxdistV - np.ones((N1)) * 10 ** (-10))
                    nxsurV = nneighforgivenr(np.concatenate((xVnextsur.reshape((N1, 1)), xembMsur), axis=1),
                                             maxdistV - np.ones((N1)) * 10 ** (-10))
                    nyFwsurV = nneighforgivenr(np.concatenate((xFM, xembMsur), axis=1),
                                               maxdistV - np.ones((N1)) * 10 ** (-10))
                    psinowM = np.full((N1, 3), np.nan)
                    psinowM[:, 0] = psi(nyFwsurV)
                    psinowM[:, 1] = psi(nxsurV)
                    psinowM[:, 2] = -psi(nwsurV)
                    cmisurV[isur + 1] = psinnei - np.mean(np.sum(psinowM, axis=1))
                icmisurV = np.argsort(cmisurV)
                rnkIc = np.where(icmisurV == 0)[0][0] + 1  # The rank of the original CMI
                pvalIc = 1 - (rnkIc - 0.326) / (nsur + 1 + 0.348)  # One-sided test
                # The corrected termination criterion
                ecC[iK] = np.vstack(
                    (ecC[iK], np.array([varind, lagind, cmiV[ind], miwV[ind], cmiV[ind] / miwV[ind]])))
                if len(iembV) == 1:
                    # This is the second embedding cycle to be tested, use only
                    # the p-value of the significance test
                    if showtxt >= 2:
                        print('%d \t %d \t %d \t %2.5f \t %2.5f \t %2.5f \n' % (ecC[iK].shape[0], ecC[iK][-1, 0],
                                                                                ecC[iK][-1, 1], ecC[iK][-1, 2],
                                                                                ecC[iK][-1, 3], ecC[iK][-1, 4]))
                    if pvalIc < alpha:
                        xembM = np.concatenate((xembM, xVnext.reshape((N1, 1))), axis=1)
                        iembV = np.append(iembV, ind)  # The index of the subsequent component is added
                    else:
                        terminator = 1
                else:
                    if showtxt >= 2:
                        print('%d \t %d \t %d \t %2.5f \t %2.5f \t %2.5f \n' % (ecC[iK].shape[0], ecC[iK][-1, 0],
                                                                                ecC[iK][-1, 1], ecC[iK][-1, 2],
                                                                                ecC[iK][-1, 3], ecC[iK][-1, 4]))
                    if len(iembV) == 2:
                        # This is the third embedding cycle to be tested, use only
                        # the p-value of the significance test
                        if pvalIc < alpha:
                            xembM = np.concatenate((xembM, xVnext.reshape((N1, 1))), axis=1)
                            iembV = np.append(iembV, ind)  # The index of the subsequent component is added
                        else:
                            terminator = 1
                    else:
                        # This is the fourth or larger embedding cycle to be
                        # tested, and terminate if p-value of the significance test
                        # is large, and if B(j)>B(j-1)>B(j-2), for B=I(y^; w | wemb) / I(y^; w, wemb)
                        # at each embedding cycle j.
                        if (pvalIc < alpha) & ((ecC[iK][-1, 4] < ecC[iK][-2, 4]) | (ecC[iK][-2, 4] < ecC[iK][-3, 4])):
                            xembM = np.concatenate((xembM, xVnext.reshape((N1, 1))), axis=1)
                            iembV = np.append(iembV, ind)  # The index of the subsequent component is added
                        else:
                            terminator = 1
            # Identify the lags of each variable in the embedding vector, if not
            # empty, and compute the R measure for each driving variable.
            if np.any((iembV < indlagM[iK, 0]) | (iembV > indlagM[iK, 1])):
                # Find the lags of the variables
                xformM = np.full((len(iembV), 2), np.nan)
                xformM[:, 0] = np.floor(iembV / Lmax)  # The variable indices
                xformM[:, 1] = np.mod(iembV + 1, Lmax)  # The lag indices for each variable
                xformM[xformM[:, 1] == 0, 1] = Lmax
                # Make computations only for the active variables, which are the
                # variables included in the mixed embedding vector.
                activeV = np.unique(xformM[:, 0]).astype(int)
                # Store the lags of the response and remove it from the active
                # variable list
                if np.intersect1d(activeV, iK).size:
                    inowV = np.where(xformM[:, 0] == iK)[0]
                    xrespM = xembM[:, inowV]
                    activeV = np.setdiff1d(activeV, iK)
                else:
                    xrespM = np.array([])  # This is the case where the response is not
                    # represented in the mixed embedding vector
                KK = len(activeV)
                indKKM = np.full((KK, 2), np.nan, dtype=int)  # Start and end in xembM of the active variables
                iordembV = np.full((len(iembV)), np.nan, dtype=int)  # the index for reordering the lag
                # matrix to set together lags of the same variable
                count = 0
                for iKK in range(KK):
                    inowV = np.where(xformM[:, 0] == activeV[iKK])[0]
                    indKKM[iKK, :] = np.array([count, count + len(inowV)])
                    iordembV[indKKM[iKK, 0]: indKKM[iKK, 1]] = inowV
                    count = count + len(inowV)
                iordembV = iordembV[:indKKM[KK - 1, 1]]
                # The total embedding vector ordered with respect to the active
                # variables and their lags, except from the response
                xembM = xembM[:, iordembV]
                # Compute the entropy for the largest state space, containing the
                # embedding vector and the future response vector. This is done
                # once for all active variables, to be used in the computation of R.
                if not xrespM.size:
                    xpastM = xembM
                else:
                    xpastM = np.concatenate((xrespM.reshape(N1, -1), xembM.reshape(N1, -1)), axis=1)
                _, dists = ANN(np.concatenate((xFM, xpastM), axis=1), nnei + 1)
                maxdistV = dists[:, -1]
                nyFV = nneighforgivenr(xFM, maxdistV - np.ones((N1)) * 10 ** (-10))
                nwV = nneighforgivenr(xpastM, maxdistV - np.ones((N1)) * 10 ** (-10))
                psi0V = psi(np.concatenate((nyFV.reshape(N1, -1), nwV.reshape(N1, -1)), axis=1))
                psinnei = psi(nnei)
                IyFw = psinnei + psi(N1) - np.mean(np.sum(psi0V, axis=1))  # I(y^T; w)
                # For each active (driving) variable build the arguments in
                # I(y^T; w^X | w^Y w^Z) and then compute it. Note that w^X is not
                # needed to be specified because it is used only to form
                # w=[w^X w^Y w^Z], which was done above one for all active variables.
                for iKK in range(KK):
                    indnowV = np.arange(indKKM[iKK, 0], indKKM[iKK, 1] + 1)
                    irestV = np.setdiff1d(np.arange(xembM.shape[1]), indnowV)
                    # Construct the conditioning embedding vector [w^Y w^Z],
                    # considering the cases one of the two components is empty.
                    if (not irestV.size) & (not xrespM.size):
                        xcondM = np.array([])
                    elif (not irestV.size) & (xrespM.size):
                        xcondM = xrespM
                    elif (irestV.size) & (not xrespM.size):
                        xcondM = xembM[:, irestV]
                    else:
                        xcondM = np.concatenate((xrespM.reshape(N1, -1), xembM[:, irestV]), axis=1)
                    # Compute I(y^T; w^X | w^Y w^Z)
                    if not xcondM.size:
                        IyFwXcond = IyFw
                    else:
                        nxFcondV = nneighforgivenr(np.concatenate((xFM, xcondM.reshape(N1, -1)), axis=1),
                                                   maxdistV - np.ones((N1)) * 10 ** (-10))
                        ncondV = nneighforgivenr(xcondM.reshape(N1, -1), maxdistV - np.ones((N1)) * 10 ** (-10))
                        psinowV = np.concatenate((psi(nxFcondV).reshape(N1, -1),
                                                  psi0V[:, 1].reshape(N1, -1), - 1 * psi(ncondV).reshape(N1, -1)),
                                                 axis=1)
                        IyFwXcond = psinnei - np.mean(np.sum(psinowV, axis=1))
                    RM[activeV[iKK], iK] = IyFwXcond / IyFw
        if ecC[iK].shape[0] > 0:
            ecC[iK] = ecC[iK][:-1, :]  # Upon termination delete tha last selected component.
    if showtxt > 0:
        print('\n')
    return RM, ecC
