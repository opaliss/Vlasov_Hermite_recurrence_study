## [Effects of Artificial Collisions, Filtering, and Nonlocal Closure Approaches on Hermite-based Vlasov-Poisson Simulations](https://arxiv.org/pdf/2412.07073)

### Abstract
Kinetic simulations of collisionless plasmas are computationally challenging due to phase space mixing and
filamentation, resulting in fine-scale velocity structures. This study compares three methods developed to reduce
artifacts related to limited velocity resolution in Hermite-based Vlasov-Poisson simulations: artificial collisions,
filtering, and nonlocal closure approaches. We evaluate each method’s performance in approximating the linear
kinetic response function and suppressing recurrence in linear and nonlinear regimes. Numerical simulations
of Landau damping demonstrate that artificial collisions, particularly higher orders of the Lenard-Bernstein
collisional operator, most effectively recover the correct damping rate across a range of wavenumbers. Moreover,
Hou-Li filtering and nonlocal closures underdamp high wavenumber modes in linear simulations, and the LenardBernstein collisional operator overdamps low wavenumber modes in both linear and nonlinear simulations. This
study demonstrates that hypercollisions offer a robust approach to kinetic simulations, accurately capturing
collisionless dynamics with limited velocity resolution.

### Python Dependencies
1. Python >= 3.9.13
2. numpy >= 1.23.3
3. matplotlib >= 3.6.0
4. scipy >= 1.7.1
5. notebook >=6.4.3
6. sympy >= 1.13.2

### Correspondence
[Opal Issan](https://opaliss.github.io/opalissan/) (Ph.D. student), University of California San Diego. email: oissan@ucsd.edu
