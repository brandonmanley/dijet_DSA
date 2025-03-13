## General info 
dsa_mc provides a Monte-Carlo implementation of the expressions for the DSA in diffractive dijet production studied in [https://inspirehep.net/literature/2843086](https://inspirehep.net/literature/2843086). 

To produce a MC sample:
```
python dsa_mc.py <sample size> <output file> <root s (GeV)> 
```

If the sample size is sufficiently large, it is usually better to run multiple instances of the script at once. In this case, one can use 
```
python submit_mc.py <output_directory> <n submissions> <sample_size> <root s (GeV)>
```
to split the sample over n files in the chosen output directory. 

Each event in the produced data has the following information:
$s, Q, x, \Delta, p_T, z, y, \phi_{kp}, \phi_{\Delta p}, \mathrm{DSA(numerator)}, \mathrm{DSA(denominator)}, \langle 1 \rangle, \langle\cos(\phi_{kp})\rangle, \langle\cos(\phi_{\Delta p})\cos(\phi_{kp})\rangle, \langle\sin(\phi_{\Delta p})\sin(\phi_{kp})\rangle$.

The angular averages are defined at the point in phase space excluding the angular variables. More details regarding the definitions of the kinematic variables can be found in [https://inspirehep.net/literature/2843086](https://inspirehep.net/literature/2843086) (and Overleaf). 

An example notebook analyzing some sample data is provided in `analysis.ipynb`.


## Initial conditions

For all of the polarized dipoles, the following form for the initial conditions is assumed:
```math
F^{(0)}(s_{10}, \eta) = a \, \eta + b \, s_{10} + c.
```
To adjust the initial conditions, vary the parameters in `dipoles/mc_ICs_random_fit.json`. N.B.: the initial conditions in `mc_ICs_random_fit.json` are a combination of random parameters for the moment amplitudes (since no data exists for these) and fitted parameters for the helicity amplitudes (taken from https://inspirehep.net/literature/2688257). 

For the unpolarized dipole amplitude, the evolved amplitudes are obtained from the code here: [https://github.com/hejajama/rcbkdipole](https://github.com/hejajama/rcbkdipole). The initial conditions are the generalized MV model and are obtained from [https://arxiv.org/abs/1012.4408](https://arxiv.org/abs/1012.4408). The generalized MV model reads
```math
  N(r^2, Y) = \frac{\sigma_0}{2}\left\{ 1- \exp \left[ \frac{\left( r^2 Q_{s0}^2 \right)^\gamma }{4} \ln \left( \frac{1}{r \Lambda_{QCD}}  + e_c e \right) \right] \right\}.
```

## Code documentation (UNDER CONSTRUCTION)

Here is a brief description of the various functions in the code. 

`plot_histogram(dfs, plot_var, weights, constraints={}, **options)`: 
Produces a histogram of `weights` in the variable `plot_var`. User can pass a dictionary of kinematic constraints as cuts on the data. 

questions + comments can be sent to manley.329@osu.edu. 
