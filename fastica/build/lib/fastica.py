
import numpy as np
import scipy.linalg as la

# whiten
def whiten(X, n_components):
    X = X - X.mean(-1)[:, None]
    u, d, _ = la.svd(X, full_matrices=False)
    V = (u / d).T[:n_components]
    X *= np.sqrt(X.shape[1])    
    return np.dot(V, X), V

# two wise choices of G
def _logcosh(x):
    # g
    gx = np.tanh(x)
    # g'
    g_x = 1 - gx ** 2
    return gx, g_x.mean(-1)

def _exp(x):
    exp = np.exp(-(x ** 2) / 2)
    # g
    gx = x * exp
    # g'
    g_x = (1 - x ** 2) * exp
    return gx, g_x.mean(axis=-1)

def decorrelation(W):
    s, u = la.eigh(np.dot(W, W.T))
    return np.dot(np.dot(u * (1. / np.sqrt(s)), u.T), W)

# fastICA
def fastICA(X, fun='logcosh', n_components=None, maxIter=200, tol=1e-04):
    X = X.T
    n, m = X.shape
    if n_components is None:
        n_components = n
    X1,V = whiten(X,n_components)
    p = float(m)
    if fun == 'logcosh':
        g = _logcosh
    elif fun == 'exp':
        g = _exp
    
    # initialize w with normal distribution
    W = np.asarray(np.random.normal(size=(n_components,n_components)))

    # calculate w iteratively
    for ii in range(maxIter):
        gwtx, g_wtx = g(np.dot(W, X1))
        W1 = decorrelation(np.dot(gwtx, X1.T) / p - g_wtx[:, None] * W)
        lim = max(abs(abs(np.diag(np.dot(W1, W.T))) - 1))
        W = W1
        if lim < tol:
            break
    S = np.dot(np.dot(W, V), X).T
    return V,W,S

if __name__ == '__main__':
    fastICA()