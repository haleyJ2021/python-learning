# intra-class coefficient correlation 组内相关系数。

#A high Intraclass Correlation Coefficient (ICC) close to 1 indicates high similarity between values from the same group
# A low ICC close to zero means that values from the same group are *not *similar


import numpy as np
def icc(Y,icc_type="icc(3,1)"):
    """
    args:
    Y:
    icc_type:
    """
    
    [n,k]=Y.shape
    
    
     # Degrees of Freedom
    dfc=k-1
    dfe=(n-1)*(k-1)
    dfr=n-1
    
    # Sum Square Total
    mean_Y=np.mean(Y)
    SST=((Y-mean_Y)**2).sum()
    
    # create the design matrix for the different levels
    x = np.kron(np.eye(k), np.ones((n, 1)))  # sessions
    x0 = np.tile(np.eye(n), (k, 1))  # subjects
    X = np.hstack([x, x0])

    # Sum Square Error
    predicted_Y = np.dot(
        np.dot(np.dot(X, np.linalg.pinv(np.dot(X.T, X))), X.T), Y.flatten("F")
    )
    residuals = Y.flatten("F") - predicted_Y
    SSE = (residuals ** 2).sum()
    MSE = SSE / dfe

    # Sum square column effect - between colums
    SSC = ((np.mean(Y, 0) - mean_Y) ** 2).sum() * n
    # MSC = SSC / dfc / n
    MSC = SSC / dfc

    # Sum Square subject effect - between rows/subjects
    SSR = SST - SSC - SSE
    MSR = SSR / dfr

    if icc_type == "icc(2,1)" or icc_type == 'icc(2,k)':
        if icc_type=='icc(2,k)':
            k=1
        ICC = (MSR - MSE) / (MSR + (k - 1) * MSE + k * (MSC - MSE) / n)
    elif icc_type == "icc(3,1)" or icc_type == 'icc(3,k)':
        if icc_type=='icc(3,k)':
            k=1
        ICC = (MSR - MSE) / (MSR + (k - 1) * MSE)
    return ICC
        
a = [[90,95,89,92,89,80,91,94,84,95],
     [89,98,89,93,91,80,94,92,82,97],
     [87,100,91,91,94,81,93,92,84,96]]
b = np.array(a)
c = icc(b.T, icc_type="icc(3,1)")
print(c)

# 0.9207797076096464
