# spsa-optimization
This repository implements Simultaneous Perturbation Stochastic Approximation (SPSA) developed by James Spall.

SPSA is a method used to finding global minima. The important stand-out feature from other optimization algorihtms is that, it approximates gradient.
ie. at each step we estimate the gradient and take a step towards the minimum. For estimation of gradient, we use only two loss measurements and hence this does not require any hand coded symbolic gradient derivations.

References:

- https://en.wikipedia.org/wiki/Simultaneous_perturbation_stochastic_approximation
- http://www.jhuapl.edu/SPSA/
