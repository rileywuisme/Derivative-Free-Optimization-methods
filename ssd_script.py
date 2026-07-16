def lambda_max(hvp, dim, rng, iters=30):
    v = rng.standard_normal(dim); v /= np.linalg.norm(v)
    lam = 0.0
    for _ in range(iters):
        w = hvp(v)
        lam = v @ w
        nw = np.linalg.norm(w)
        if nw == 0: break
        v = w / nw
    return abs(lam)

def linesearch(f, xk, pk, derphi0, old_fval=None, alpha0 = 1.0, c1=1e-4, amin=1e-6):
    '''Parameters: 
    f: objective function
    xk: starting point (ndarray)
    pk: search direction (ndarray)
    old_fval: f(xk)
    c1: armijo condition constant
    amin: minimum alpha since there's no curvature condition. 

    return: 
    alpha: new step
    fc: # of function evaluation
    phi1: f value at alpha
    '''
    
    fc = [0]
    def phi(alpha):
        fc[0] += 1
        return f(xk + alpha*pk)

    if old_fval is None: 
        phi0 = phi(0.)
    else:
        phi0 = old_fval

    alpha, phi1 = scalar_search_armijo(
            phi, phi0, derphi0, c1=c1, alpha0=alpha0, amin=amin)

    return alpha, fc[0], phi1

#input alpha and l, independent variables
#0 < alpha < 2l/(d*lambda) 

def ssd(f, gradf, x, l = None, ls=None, max_iter=500, seed=0, amin = 1e-6):
    '''
    followed algorithm 2.1:
    1: Inputs: α, l. step size, subspace rank
    2: Initialize: x0 . arbitrary initialization
    3: for k = 1, 2, . . . do
    4: Generate P_k
    5: Apply recursion

    updating the linesearch method

    Parameters:
    f: objective functio
    x: starting point x_0
    l: subspace dimension
    alpha = whether or not to implement linesearch

    Return:
    fx: a list of history of function evaluations
    x_history: a list of history of x values
    '''

    #return lists
    fx_history = []
    x_history = []
    stepsize_history = []

    #determining the subspace dimension
    dim = len(x)
    if l is None:
        # default choice for subspace:
        l = min([max([5, int(np.sqrt(dim)) ]), dim])
    if l > dim: raise ValueError('subspace dim should be < dimension')
    if l == dim: print(' Are you sure you want l == dim?')

    
    fval = float(f(x))
    rng = np.random.default_rng(seed)
    alpha0 = float(l)/dim

    #computing optimal stepsize for fixed step
    hvp = lambda v: p.hprod(v, x=x)
    L = lambda_max(hvp, dim, np.random.default_rng(seed + 1))
    alpha_nols = l / (dim * L)
    
    for n in range(max_iter):
        grad = gradf(x)

        if l == dim: #simply gradient descent
            Q = np.eye(dim)
            pk = y = grad
        else:
            Z = rng.standard_normal((dim, l))
            Q, _ = linalg.qr(Z, mode='economic') #economic mode for finding the QR so it's not a full square matrix
            P = np.sqrt(dim / l) * Q 
            y = P.T @ grad
            pk = - P @ y

        if ls == 'on':
            derphi0 = -float(y @ y) #or could be written with pk with a factor l/d. 
            alpha_ls, nfev, phi1 = linesearch(f, x, pk, derphi0, old_fval=fval, alpha0=max(alpha0, amin), amin=amin)
            #if search failed
            if alpha_ls is None:
                alpha_ls = 1e-8
                phi1 = old_fval + alpha_ls*derphi0
                alpha0 = alpha_ls
            else:
                alpha0 = 2.2 * alpha_ls #otherwise the armijo algo simply return alpha0
            stepsize_history.append(alpha_ls)
            x = x + alpha_ls * pk
            fval = phi1
        elif ls == 'off':
            x = x + alpha_nols * pk
            fval = float(f(x))
        else:
            raise ValueError ("Please choose linesearch mode.")
        
        #plug x into function
        fx_history.append(fval)
        x_history.append(np.array(x))

    return np.array(fx_history), np.array(x_history), np.array(stepsize_history)