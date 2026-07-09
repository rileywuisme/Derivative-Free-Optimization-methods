# Derivative-Free-Optimization-methods
Summer 2026 working with Prof. Stephen Becker on derivative free optimization methods. Starting with understanding stochastic subspace descent, a method similar to stochastic gradient descent. 
Stochastic Subspace Descent: 

$$\vec{x}_{k+1} = \vec{x}_k + \alpha \textbf{P}_k \textbf{P}_k^T \nabla f(\vec{x_k})$$

$P \in R^{d \times l}, l \leq d$.  

We apply this method specifically to nonlinear least squares problems, where we have the new normal equation 
$$-(JP)^\top r = (JP)^\top (JP) n$$