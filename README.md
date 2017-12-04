# spsa-optimization
This repository implements Simultaneous Perturbation Stochastic Approximation (SPSA) developed by James Spall.

SPSA is a method used to find global minima. The important stand-out feature from other optimization algorihtms is that, it approximates gradient instead of computing the gradient. At each step we estimate the gradient and take a step towards the minimum. For estimation of gradient, we use only two loss measurements and hence this does not require any hand coded symbolic gradient derivations.


For the usage of this optimizer, refer to [scripts/tests/test_spsa_multi_linear.py](https://github.com/rajcscw/spsa-optimization/blob/master/scripts/tests/test_spsa_multi_linear.py).

References:

- https://en.wikipedia.org/wiki/Simultaneous_perturbation_stochastic_approximation
- http://www.jhuapl.edu/SPSA/
