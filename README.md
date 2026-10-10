# Derivative-Free-Optimization-methods
Summer 2026 working with Prof. Stephen Becker on derivative free optimization methods. Starting with understanding stochastic subspace descent, a method similar to stochastic gradient descent. 
Stochastic Subspace Descent: 

$$\vec{x}_{k+1} = \vec{x}_k + \alpha \textbf{P}_k \textbf{P}_k^T \nabla f(\vec{x_k})$$

$P \in R^{d \times l}, l \leq d$.  

We apply this method specifically to nonlinear least squares problems, where we have the new normal equation 
$$-(JP)^\top r = (JP)^\top (JP) n$$

Additionally, we combined the SSD with SAGA (Defazio et al.) 

The algorithm is as follows: 
$$
    x_{k+1} = x - \alpha_1 (I-PP^T)z - \alpha_2 PP^T \nabla f(x_k)
$$
where z is a running sum approximating $\nabla f(x)$ defined by 
$$ z = z + PP^T (\nabla f(x_k) - z) $$
It is an analogue to the original SAGA (though we are not restricted to finite sum objectives)
$$x_{k+1} = x_k - \alpha \left [ f'_j(x_k) - f'_j(x_{k-1}) + \frac{1}{n} \sum_{i=1}^n f'_i(x_{k-1}) \right ]$$
Notice the SAGA algorithm relies on one step size $\alpha$ for every iteration even though there's an imbalance of how the step size is applied. 
