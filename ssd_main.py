def estimate_L(p, iters=30, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(p.n); v /= np.linalg.norm(v)
    lam = 0.0
    for _ in range(iters):
        w = p.hprod(v, x=p.x0)
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

def ssd(f, gradf, x, l = None, ls=None, max_iter=500, seed=0, amin = 1e-6, L=None):
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

    if ls == 'off' and L is None:
        raise ValueError("ls='off' requires L")
    if ls == 'off':
        alpha_nols = l / (dim * max(L, 1e-16))
    
    for n in range(max_iter):
        grad = gradf(x)

        if l == dim: #simply gradient descent
            y = grad
            pk = -grad
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
                alpha_ls, phi1 = 0.0, fval    # null step; fresh subspace next iteration
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

####################################################################################

# Gauss-Newton SSD
def gn_ssd(x, cons, l=None, ls=None, max_iter=200, seed=0, amin=1e-10):
    '''
    input: x is the starting point; P is dxl; r the residual function
    Q has Haar distribution
    Assume alpha = 1
    iteration: x = P
    '''

    dim = len(x)
    rng = np.random.default_rng(seed)

    #return variables
    x_history = []
    fx_history = []
    stepsize_history = []

    #subspace dimension
    dim = len(x)
    if l is None:
        # default choice for subspace:
        l = min([max([5, int(np.sqrt(dim)) ]), dim])
    if l > dim: raise ValueError('subspace dim should be < dimension')
    if l == dim: print(' Are you sure you want l == dim?')

    #problem setup 
    def fobj(y):                      # scalar objective for the line search
        ry = cons(y)                  # residual only, no Jacobian
        return 0.5 * np.dot(ry, ry)

    alpha0 = 1.0
    
    for n in range(max_iter):
        r , J = cons(x, gradient=True) #residual and Jacobian
        fval = 0.5 * np.dot(r, r)
        
        Z = rng.standard_normal((dim, l))
        Q, _ = linalg.qr(Z, mode='economic') #economic mode for finding the QR so it's not a full square matrix

        #solving for the step
        J_tild = J @ Q
        pk = np.linalg.lstsq(J_tild, - r, rcond=None)[0]

        if ls == 'on':
            derphi0 = (J_tild.T @ r) @ pk
            direc = Q @ pk
            alpha_ls, nfev, phi1 = linesearch(fobj, x, direc, derphi0, old_fval=fval,alpha0=max(alpha0, amin), amin=amin)
            #if it fails
            if alpha_ls is None: 
                alpha_ls, phi1 = 0.0, fval
            else:
                alpha0 = 2.5 * alpha_ls
            stepsize_history.append(alpha_ls)
            fval = phi1
            x = x + alpha_ls * direc
        elif ls == 'off': #alpha = 1
            x = x + Q @ pk
            fval = float(fval)
        else:
            raise ValueError("please choose linesearch mode.")

        x_history.append(np.array(x))
        fx_history.append(fval)

    return np.array(fx_history), np.array(x_history), np.array(stepsize_history)