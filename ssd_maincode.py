#packages
from scipy import linalg
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import approx_fprime
import pycutest
import time

VALID_LS = ('armijo', 'polyak', 'BB', 'off', 'saga0', 'saga1', 'saga2', 'saga3')

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

def armijo_backtrack(phi, phi0, derphi0, alpha0, c1=1e-4, amin=1e-6, lo=0.1, hi=0.5, max_evals=60):
    '''
    Safeguarded backtracking Armijo search (Nocedal & Wright, Sec. 3.5). Improved over SciPy's 
    scalar_search_armijo. 

    A rejected trial alpha is replaced by the minimizer of the quadratic interpolating
    phi(0), phi'(0), phi(alpha), restricted to [lo*alpha, hi*alpha]. The interval prevents the
    step from collapsing when phi grows faster than quadratically (quartic objectives
    with an overshooting alpha0 that result in phi(alpha) being infinity), 
    which scipy's scalar_search_armijo does not prevent.

    return: (alpha, phi(alpha)) with alpha >= amin, or (None, phi0) on failure.
    '''
    if not derphi0 < 0:  # not a descent direction (also catches NaN)
        return None, phi0
    alpha = alpha0
    for _ in range(max_evals):
        if alpha < amin:
            break
        phi_a = phi(alpha)
        if np.isfinite(phi_a) and phi_a <= phi0 + c1 * alpha * derphi0: #armijo condition
            return alpha, phi_a
        if np.isfinite(phi_a):
            # denominator > 0 whenever the Armijo test fails and derphi0 < 0
            # quadratic interpolation to find a new alpha
            alpha_q = -derphi0 * alpha**2 / (2.0 * (phi_a - phi0 - derphi0 * alpha))
        else: #phi_a is infinite
            alpha_q = lo * alpha
        #check the bound: [0.1alpha, 0.5alpha]. AI's idea. 
        alpha = min(max(alpha_q, lo * alpha), hi * alpha)
    return None, phi0

def linesearch(f, xk, pk, derphi0, old_fval=None, alpha0 = 1.0, c1=1e-4, amin=1e-6):
    '''Parameters: 
    f: objective function
    xk: starting point (ndarray)
    pk: search direction (ndarray)
    derphi0: directional derivative of f at xk along pk (must be < 0)
    old_fval: f(xk)
    c1: armijo condition constant
    amin: minimum alpha since there's no curvature condition. 

    return: 
    alpha: new step (None if the search failed)
    fc: # of function evaluation
    phi1: f value at alpha (f(xk) if the search failed)
    '''
    fc = [0]
    def phi(alpha):
        fc[0] += 1
        return f(xk + alpha*pk)

    phi0 = phi(0.) if old_fval is None else old_fval
    alpha, phi1 = armijo_backtrack(phi, phi0, derphi0, alpha0, c1=c1, amin=amin)
    return alpha, fc[0], phi1

#forward different to calculate directional derivative
def finite_diff_directional_deriv(x, f, fx, P, h=1e-8):
    '''h should be the same as c in spsa'''
    return np.array([(f(x+h*p) - fx)/h for p in P])

def dir_deriv(f, x, fx, v, h=1e-8):
    '''Forward-difference estimate of grad f(x) @ v, differenced along the unit vector v/||v||
    so the finite-difference error does not depend on ||v||. Costs 1 evaluation.'''
    nv = np.linalg.norm(v)
    if nv == 0:
        return 0.0
    return (f(x + (h / nv) * v) - fx) / h * nv

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
def double_ls(func, phi0, pk, x, y, z, x0, v1, v2, h, d2, debug=False):
    '''
    Quadratic model of phi(eta1, eta2) = func(x0 - eta1*v1 - eta2*v2), fitted from
    phi(0,0), the gradient of phi at 0, and the stepsize values phi(x,0), phi(0,y), phi(x,y).

    func: function
    phi0: last function evaluation, func(x0)
    pk: -P P.T gradF (so v1 = -pk)
    x, y: stepsize values of eta1, eta2 (y may be negative)
    z: continuously updating gradient (v2 = z - P P.T z)
    h: precision of finite difference

    Cost: 5 function evaluations.
    return: (eta, phi(eta)) if A is positive definite and phi(eta) < phi0, else (None, None)
    '''
    def phi(eta1, eta2):
        return func(x0 - eta1 * v1 - eta2 * v2)
    phi00 = phi0
    b1 = - pk @ pk # d phi/d eta1 at 0
    b2 = - d2
    phix0 = phi(x, 0) #+1
    phi0y = phi(0, y) #+1
    phixy = phi(x, y) #+1

    c = phi00
    a11 = 2*(phix0 - b1 * x - c) / (x**2)
    a22 = 2*(phi0y - b2 * y - c) / (y**2)
    a12 = (phixy - 1/2 * (a11 * x**2 + a22 * y**2) - b1 * x - b2 * y - c) / (x*y)

    A = np.array([[a11, a12], [a12, a22]])
    b = np.array([b1, b2])
    if not np.all(np.isfinite(A)):
        if debug: print("double_ls: non-finite entry in A")
        return None, None
    # Sylvester test: the model has a minimizer only if A is positive definite
    if a11 <= 0 or a11 * a22 - a12**2 <= 0:
        if debug: print(f"double_ls: A not positive definite, eigenvalues {np.linalg.eigvalsh(A)}")
        return None, None
    if np.linalg.cond(A) > 1e12:
        if debug: print("double_ls: A ill-conditioned")
        return None, None

    new_eta = np.linalg.solve(A, -b) #newton one step for quadratic
    phi1 = phi(new_eta[0], new_eta[1]) #+1
    if not phi1 < phi00: #next step would not decrease f
        return None, None
    return new_eta, phi1

def ls_trial_steps(f, x, fval, pk, PT_times_grad, v2, alpha0, amin, h):
    '''
    1D Armijo searches along v1 = -pk and along the descent sign of v2 (saga1, saga3).
    Convention: trial points are x - eta1*v1 and x - eta2*v2; eta2 may be negative.
    A failed search returns None for its eta.

    Along v1 = P P^T grad f the optimal multiplier is >= 1/lambda_max(P^T H P), so a multiplier
    floor amin is harmless. Along v2 the optimal multiplier -grad f @ v2 / (v2^T H v2) can be
    arbitrarily small (v2 is often nearly orthogonal to grad f), so amin is applied to the
    displacement ||eta2 * v2|| instead.
    '''
    derphi0 = -float(PT_times_grad @ PT_times_grad)
    eta1, _, phi_e1 = linesearch(f, x, pk, derphi0, old_fval=fval, alpha0=max(alpha0, amin), amin=amin)
    d2 = dir_deriv(f, x, fval, v2, h) # grad f(x) @ v2   #+1
    eta2, phi_e2 = None, None
    if d2 != 0:
        sgn = -np.sign(d2) # sgn*v2 is a descent direction, slope -|d2|
        amin2 = amin / np.linalg.norm(v2)
        a2, _, phi_a2 = linesearch(f, x, sgn * v2, -abs(d2), old_fval=fval, alpha0=max(alpha0, amin2), amin=amin2)
        if a2 is not None:
            eta2, phi_e2 = -sgn * a2, phi_a2 # x + a2*sgn*v2 = x - eta2*v2
    return eta1, phi_e1, eta2, phi_e2, d2

def ssd(obj, x, ls=None, gradf=None, l = None, max_fn_call = 7000, seed=0, amin = 1e-6, L=None, 
        alpha_max=10, qr = 'cholesky', h=1e-8, eta_init=0.005, min_trial_step_norm=1e-5, z_init = None, debug=False):
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
    if ls not in VALID_LS:
        raise ValueError(f"unknown ls={ls!r}; expected one of {VALID_LS}")

    #initialize function call wrapper
    f = counted(obj)
    x = np.array(x, dtype=float)

    #return lists
    fx_history = []
    x_history = []
    stepsize_history = []
    fn_call_history = [] #for performance profile
    counters = {'ls_failed': 0, 'dls_rejected': 0, 'one_dir': 0, 'both_failed': 0}

    def record(step):
        x_history.append(np.array(x))
        fx_history.append(fval)
        fn_call_history.append(f.calls[0])
        stepsize_history.append(step)

    #determining the subspace dimension
    dim = len(x)
    if l is None:
        # default choice for subspace:
        l = min([max([5, int(np.sqrt(dim)) ]), dim])
    if l > dim: raise ValueError('subspace dim should be <= dimension')
    if l == dim: print(' Are you sure you want l == dim?')

    is_saga = ls.startswith('saga')
    variant = ls[4:] if is_saga else None
    zero_step = [0.0, 0.0] if is_saga else 0.0

    fval = float(f(x))
    rng = np.random.default_rng(seed)
    alpha0 = float(l)/dim
    record(zero_step) # f(x0) at fn_call = 1

    #fixed step: P is orthonormal, so the natural step is 1/L
    if ls == 'off':
        if L is None:
            raise ValueError("ls='off' requires L")
        alpha_nols = 1.0 / max(L, 1e-16)

    #for saga algo
    if is_saga:
        z = np.zeros(dim) #when z is useless
        if z_init == 'grad': #when z is useful
            for i in range(dim): #+dim
                x_pert = x.copy()
                x_pert[i] += h
                z[i] = (f(x_pert) - fval) / h
            record(zero_step) # record the cost of the gradient initialization

    first_step = True
    new_eta = None   # previous (eta1, eta2); None means (re)initialize the stepsize
    x_prev, fval_prev = None, None   # for BB

    while f.calls[0] <= max_fn_call:      

        #first saga step when z is the full gradient: Armijo step along -z
        if is_saga and first_step and np.any(z):
            alpha, _, phi1 = linesearch(f, x, -z, -float(z @ z), old_fval=fval, alpha0=max(alpha0, amin), amin=amin)
            if alpha is None:
                counters['ls_failed'] += 1
                alpha = 0.0
            else:
                x = x - alpha * z
                fval = phi1
            first_step = False
            record([alpha, 0.0])
            continue
        
        if l == dim: #simply gradient descent
            if gradf is None:
                raise ValueError(f"l == dim == {dim} requires gradf (dim too small for sketching)")
            P = np.eye(dim)
            PT_times_grad = gradf(x)
            pk = -PT_times_grad
        else:
            Z = rng.standard_normal((dim, l))
            if qr == 'scipy':
                P, _ = linalg.qr(Z, mode='economic', check_finite=False, overwrite_a=True)
            elif qr == 'cholesky': #cholesky reduce the time by a factor of 6   
                R = np.linalg.cholesky(Z.T @ Z)
                P = linalg.solve_triangular(R, Z.T, lower=True).T
            else:
                raise ValueError(f"unknown qr={qr!r}; expected 'scipy' or 'cholesky'")

            #define P_times_grad
            if gradf is None: #the way to calculate jvp compatible with CUTEst
                PT_times_grad = finite_diff_directional_deriv(x,f,fval, P.T, h=h)
            else: #cheating
                PT_times_grad = P.T @ gradf(x)
            pk = - P @ PT_times_grad

        #first saga step when z is zero: Armijo step along pk, then z = P P^T grad f
        if is_saga and first_step:
            derphi0 = -float(PT_times_grad @ PT_times_grad)
            alpha, _, phi1 = linesearch(f, x, pk, derphi0, old_fval=fval, alpha0=max(alpha0, amin), amin=amin)
            if alpha is None:
                counters['ls_failed'] += 1
                alpha = 0.0
            else:
                x = x + alpha * pk
                fval = phi1
            z = -pk # now informative so double_ls has a real v2 next iter
            first_step = False
            record([alpha, 0.0])
            continue
        
        if ls == 'armijo':
            derphi0 = -float(PT_times_grad @ PT_times_grad)
            alpha_ls, nfev, phi1 = linesearch(f, x, pk, derphi0, old_fval=fval, alpha0=max(alpha0, amin), amin=amin)
            if alpha_ls is None: #if search failed
                counters['ls_failed'] += 1
                alpha_ls = 0.0
            else:
                if nfev <= 1:
                    # the alpha0 was probably too conservative since it was accepted right away
                    alpha0 = min(2.2 * alpha_ls, alpha_max) # aggressive increase
                else:
                    alpha0 = alpha_ls
                x = x + alpha_ls * pk
                fval = phi1
            record(alpha_ls)

        elif ls == 'polyak':
            #assumes NLS problem format
            fmin = 0 #otherwise it's too much work to compute f*
            eta = (fval - fmin) / (PT_times_grad @ PT_times_grad)
            alpha_pol = min(alpha_max, eta)
            x = x + alpha_pol * pk
            fval = float(f(x))
            record(alpha_pol)

        elif ls == 'BB':
            alpha_bb = alpha0
            if x_prev is not None:
                s = x - x_prev
                if np.any(s):
                    sty = dir_deriv(f, x, fval, s, h) - dir_deriv(f, x_prev, fval_prev, s, h) #+2
                    if sty > 0:
                        alpha_bb = min(alpha_max, (s @ s) / sty) #eta can blow up if s ortho to y
            x_prev, fval_prev = x, fval
            x = x + alpha_bb * pk
            fval = float(f(x))
            record(alpha_bb)

        elif ls == 'off':
            x = x + alpha_nols * pk
            fval = float(f(x))
            record(alpha_nols)

        else: # saga0 - saga3
            v1 = - pk
            v2 = z - P @ (P.T @ z)

            #step norms
            n1 = max(np.linalg.norm(v1), np.finfo(float).tiny) 
            n2 = np.linalg.norm(v2)
            eta1 = eta2 = phi_e1 = phi_e2 = None
            step = [0.0, 0.0]

            if n2 <= 1e-12 * n1:
                #v2 adds no direction: 1D Armijo step along pk
                use_2d = False
                derphi0 = -float(PT_times_grad @ PT_times_grad)
                eta1, _, phi_e1 = linesearch(f, x, pk, derphi0, old_fval=fval, alpha0=max(alpha0, amin), amin=amin)
            else:
                use_2d = True
                if variant == '1' or (variant == '3' and new_eta is None):
                    eta1, phi_e1, eta2, phi_e2, d2 = ls_trial_steps(f, x, fval, pk, PT_times_grad, v2, alpha0, amin, h)
                elif variant == '0' or new_eta is None: # saga0, or saga2 (re)initialization normalized
                    eta1, eta2 = eta_init / n1, eta_init / n2
                else: # saga2, saga3: stepsize from the previous step
                    eta1 = max(abs(2*new_eta[0]), min_trial_step_norm / n1)
                    eta2 = max(abs(2*new_eta[1]), min_trial_step_norm / n2)


            if use_2d and eta1 is not None and eta2 is not None:
                eta, phi1 = double_ls(f, fval, pk, eta1, eta2, z, x, v1, v2, h, d2, debug=debug)
                if eta is None:
                    counters['dls_rejected'] += 1
                    new_eta = None # re-initialize the stepsize next iteration
                else:
                    x = x - eta[0] * v1 - eta[1] * v2
                    fval = phi1
                    new_eta = eta
                    step = [eta[0], eta[1]]
            elif eta1 is not None:
                # only the v1 direction succeeded (or v2 degenerate): 1D step along v1
                x = x - eta1 * v1
                fval = phi_e1
                new_eta = [eta1, 0.0]
                step = [eta1, 0.0]
                if use_2d: counters['one_dir'] += 1
            elif eta2 is not None:
                # only the v2 direction succeeded: 1D step along v2
                x = x - eta2 * v2
                fval = phi_e2
                new_eta = [0.0, eta2]
                step = [0.0, eta2]
                counters['one_dir'] += 1
            else:
                counters['both_failed'] += 1
                new_eta = None

            # SAGA update at x_k, applied whether or not a step was taken:
            # z <- (I - P P^T) z + P P^T grad f(x_k)
            z = v2 - pk
            record(step)

    return np.array(x_history), np.array(fx_history), np.array(stepsize_history), np.array(fn_call_history), counters