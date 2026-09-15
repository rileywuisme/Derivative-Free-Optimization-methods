#packages
from scipy import linalg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize._linesearch import scalar_search_armijo
from scipy.optimize import approx_fprime
from scipy.optimize._linesearch import line_search_armijo
import pycutest
import time

#local L, only global for quadratic. Problem for SSD
def estimate_L(hvp, dim, seed=0, iters=30):
    v = np.random.default_rng(seed).standard_normal(dim); v /= np.linalg.norm(v)
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

    differ from scipy: derphi0 as argument. 
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

#forward different to calculate directional derivative
def finite_diff_directional_deriv(x, f, fx, P, h=1e-8):
    '''h should be the same as c in spsa'''
    return np.array([(f(x+h*p) - fx)/h for p in P])

#count function evaluations
def counted(fn):
    calls = [0]
    def wrapped(*args, **kwargs):
        calls[0] += 1
        return fn(*args, **kwargs)
    wrapped.calls = calls
    wrapped.reset = lambda: calls.__setitem__(0, 0)
    return wrapped

#double linesearch
def double_ls(func, phi0, pk, x, y, z, x0, v1, v2, h):
    '''
    func: function
    phi0: last function evaluation
    pk: P P.T gradF 
    x, y: two variables in phi
    z: continuously updating gradient
    v1, v2: two directions in phi
    h: precision of finite difference
    
    f00: f(x) = phi(0, 0)
    pk = Q Q.T gradf
    c = coefficients of the quadratic
    '''
    def phi(eta1, eta2):
        return func(x0 - eta1 * v1 - eta2 * v2)
    phi00 = phi0
    delphi1 = - pk @ pk
    zTgradf = (func(x0 + h*z) - phi00 )/ h #+1
    delphi2 = - zTgradf - z.T @ pk #pk is negative
    phix0 = phi(x, 0) #+1
    phi0y = phi(0, y) #+1
    phixy = phi(x, y) #+1

    c = phi00
    b1 = delphi1
    b2 = delphi2
    #task: look for ways to limit numerical error in the coeficient calculations
    a11 = 2*(phix0 - delphi1 * x - phi00) / (x**2)
    a22 = 2*(phi0y - delphi2 * y - phi00) / (y**2)
    a12 = (phixy - 1/2 * (a11 * x**2 - a22 * y**2) - b1 * x - b2 * y - c) / (x*y)

    A = np.array([[a11, a12], [a12, a22]])
    if not np.all(np.isfinite(A)):
        return None, None
    if np.linalg.cond(A) > 1e12:
        return None, None
    b = np.array([b1, b2])

    f = lambda eta: (1/2)*eta.T @ A @ eta + b.T @ eta + c
    new_eta = np.linalg.solve(A, -b) #newton one step for quadratic
    phi1 = phi(new_eta[0], new_eta[1]) #+1
    if phi1 >= phi00: #means next step is going to increase instead of decrease
        return None, None
    # print(np.linalg.eigvals(A))
    return new_eta, phi1

def ssd(obj, x, ls=None, gradf=None, l = None, max_fn_call = 7000, seed=0, amin = 1e-6, L=None, 
        alpha_max=10, qr = 'cholesky', h=1e-8, eta_init=0.005, z_init = None):
    '''
    Parameters:
    f: objective function
    x: starting point x_0
    l: subspace dimension
    ls: linesearch 'on' or 'off'
    max_iter: maximum iterations
    seed: random seed for rng
    amin: minimum step size
    L: Lipschitz value for calculating best alpha for fixed step version
    alpha_max: maximum step size
    qr: calculating qr decomposition using scipy.linalg.qr 'scipy' or 'cholesky' (default)
    alpha = whether or not to implement linesearch
    h: the denominator for all finite difference schemes

    Return:
    fx: an array of history of function evaluations
    x_history: an array of history of x values
    stepsize_history: an array of history of stepsize

    7/21/2026 note: change from max_iter to max_fn_call
    8/22/2026 note: for comparison purposes have saga0 as the original saga inspired algorithm with only eta_max as the input
                    saga1 does 1D linesearch in both directions, input x = alpha1, y = alpha2, 
                    (alpha is the stepsize) then double_ls to find the minimum. 
                    saga2 has an initialized eta1 and eta2, then update the eta with the previous eta
                    saga3 is similar to saga2, but the initialized etas are from 1d linesearch. A combo of saga1 and saga2.
    '''
    #initialize function call wrapper
    f = counted(obj)

    #return lists
    fx_history = []
    x_history = []
    stepsize_history = []
    fn_call_history = [] #for performance profile

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
    fval_prev = fval

    #for saga algo
    if ls.startswith('saga'):
        z = np.zeros(dim) #when z is useless
        if z_init == 'grad': #when z is useful
            for n in range(dim): #+1
                x_pert = x.copy()
                x_pert[n] += h
                z[n] = (f(x_pert) - fval) / h

    #computing best alpha for fixed stepsize
    if ls == 'off' and L is None:
        raise ValueError("ls='off' requires L")
    if ls == 'off':
        alpha_nols = l / (dim * max(L, 1e-16))

    first_step = True
    new_eta = [eta_init, eta_init]
    failed_one = 0
    failed_both = 0
    
    while f.calls[0] <= max_fn_call:      

        #first step "saga":
        #if z is useful, skip the following steps and find a stepsize using armijo/polyak
        #if z is useless, do the following steps and still use armijo/polyak
        if ls.startswith('saga') and first_step and np.any(z):
            pk = -z
            derphi0 = -float(z @ z)
            alpha, _, phi1 = linesearch(f, x, pk, derphi0, old_fval=fval, alpha0=max(alpha0, amin), amin=amin)
            x = x + (alpha or 0.0) * pk
            fval = phi1 if alpha is not None else fval
            first_step = False
            # z stays the gradient → v2 = z - P Pᵀ z is nonzero next iter, double_ls works
            # record history, continue
            continuex
        
        if l == dim: #simply gradient descent
            if gradf is None:
                raise ValueError(f"l == dim == {dim} requires gradf (dim too small for sketching)")
            grad = gradf(x)
            PT_times_grad = grad
            pk = -grad
            print('l = dim, gradient descent mode, SSD not engaged. ')
        else:
            Z = rng.standard_normal((dim, l))
            if qr == 'scipy':
                P, _ = linalg.qr(Z, mode='economic', check_finite=False, overwrite_a=True)
            elif qr == 'cholesky': #cholesky reduce the time by a factor of 6   
                G = Z.T @ Z
                R = np.linalg.cholesky(G)
                P = linalg.solve_triangular(R, Z.T, lower=True).T
                if ls == 'off':
                    P = np.sqrt(dim / l) * Q #just for fixed step bc Lipchitz is known
            #put the scaling in the stepsize for fixed step

            #define P_times_grad
            if gradf is None: #the way to calculate jvp compatible with CUTEst
                PT_times_grad = finite_diff_directional_deriv(x,f,fval, P.T, h=h)
            else: #cheating
                grad = gradf(x)
                PT_times_grad = P.T @ grad
            pk = - P @ PT_times_grad

        if ls.startswith('saga') and first_step:
            derphi0 = -float(PT_times_grad @ PT_times_grad)
            alpha, _, phi1 = linesearch(f, x, pk, derphi0, old_fval=fval, alpha0=max(alpha0, amin), amin=amin)
            x = x + (alpha or 0.0) * pk
            fval = phi1 if alpha is not None else fval
            z = -pk # now informative so double_ls has a real v2 next iter
            first_step = False
            # record history, continue
            continue
        
        if ls == 'armijo':
            derphi0 = -float(PT_times_grad @ PT_times_grad)
            alpha_ls, nfev, phi1 = linesearch(f, x, pk, derphi0, old_fval=fval, alpha0=max(alpha0, amin), amin=amin)
            if alpha_ls is None: #if search failed
                alpha_ls, phi1 = 0.0, fval
            else:
                # alpha0 = min(2.2 * alpha_ls, alpha_max) #otherwise the armijo algo simply return alpha0
                if nfev <= 1:
                    # the alpha0 was probably too conservative since it was accepted right away
                    alpha0 = min(2.2 * alpha_ls, alpha_max) # aggressive increase
                else:
                    alpha0 = alpha_ls # or consider a *small* increase
            stepsize_history.append(alpha_ls)
            x = x + alpha_ls * pk
            fval = phi1
        elif ls == 'polyak':
            #assumes NLS problem format
            #need to do something about alpha_max
            fmin = 0 #otherwise it's too much work to compute f*
            eta = (fval - fmin) / (PT_times_grad @ PT_times_grad)
            alpha_pol = min(alpha_max, eta)
            x = x + alpha_pol * pk
            fval = float(f(x))
            stepsize_history.append(alpha_pol)
        elif ls == 'BB':
            if not x_history:
                alpha_bb = alpha0
            else:
                x_1 = x_history[-1]
                s = x - x_1
                gradf_k = (f(x+h*s) - fval) / h
                gradf_k_1 = (f(x_1+h*s) - fval_prev) / h
                sty = gradf_k - gradf_k_1
                if sty <= 0:
                    alpha_bb = alpha0
                else:
                    alpha_bb = min(alpha_max, (s@s)/sty) #eta can blow up if s ortho to y
            x = x + alpha_bb * pk
            fval_prev = fval
            fval = float(f(x))
            stepsize_history.append(alpha_bb)
        elif ls.startswith('saga'):
            v1 = - pk
            v2 = z - P @ (P.T @ z)

            variant = ls[4:] #the number
            if variant == '0':
                eta1 = eta2 = eta_init
            if variant == '1':
                derphi0 = -float(PT_times_grad @ PT_times_grad)
                #eta1 and eta2 could be None (part of scipy's algo), need safeguards.
                eta1, _, _ = linesearch(f, x, pk, derphi0, old_fval=fval, alpha0=max(alpha0, amin), amin=amin)
                derphi0_v2 = (f(x + h*v2) - fval) / h
                eta2, _, _ = linesearch(f, x, -v2, derphi0_v2, old_fval=fval, alpha0=max(alpha0, amin), amin=amin)
            if variant == '2' or variant == '3':
                eta1 = max(abs(2*new_eta[0]), 1e-8)
                eta2 = max(abs(2*new_eta[1]), 1e-8)

            if eta1 is not None and eta2 is not None:
                new_eta, phi1 = double_ls(f, fval, pk, eta1, eta2, z, x, v1, v2, h)
                if new_eta is None: #safeguard from double_ls return None
                    new_eta = [eta_init, eta_init]
                    x_history.append(np.array(x))
                    fx_history.append(fval)
                    fn_call_history.append(f.calls[0])
                    continue
            elif eta1 is not None:
                # only v1 direction succeeded, 1D step along v1
                new_eta = [eta1, 0.0] #no safeguard for ascend
                failed_one += 1
            elif eta2 is not None:
                # only v2 direction succeeded, 1D step along v2
                new_eta = [0.0, eta2]
                failed_one += 1
            else:
                new_eta = None
                x_history.append(np.array(x))
                fx_history.append(fval)
                fn_call_history.append(f.calls[0])
                failed_both = 0
                continue
            x = x - new_eta[0] * v1 - new_eta[1] * v2
            z = v2 - pk
            fval = float(f(x))
            stepsize_history.append(new_eta)
                
        elif ls == 'off':
            x = x + alpha_nols * pk
            fval = float(f(x))
            stepsize_history.append(alpha_nols)
        else:
            raise ValueError(f"unknown ls={ls!r}; expected 'armijo', 'polyak', 'BB', 'off', or None")

        x_history.append(np.array(x))
        fx_history.append(fval)
        fn_call_history.append(f.calls[0])

    return np.array(x_history), np.array(fx_history), np.array(stepsize_history), np.array(fn_call_history), failed_one + failed_both