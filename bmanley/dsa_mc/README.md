DSA_MC provides a Monte-Carlo implementation to produce pseudo-data from the expressions for the DSA in diffractive dijet production studied in [https://inspirehep.net/literature/2843086](https://inspirehep.net/literature/2843086). 

To run the code: 
```
python dsa_mc.py <sample size> <output file> <root s (GeV)> 
```

To adjust the initial conditions, vary the parameters in `dipoles/mc_ICs_random_fit.json`. N.B.: the initial conditions in `mc_ICs_random_fit.json` are a combination of random parameters for the moment amplitudes (since no data exists for these) and fitted parameters for the helicity amplitudes (taken from https://inspirehep.net/literature/2688257). 

For all of the polarized dipoles, the following form for the initial conditions is assumed:
```math
F^{(0)}(s_{10}, \eta) = a \, \eta + b \, s_{10} + c.
```
For the unpolarized dipole amplitude, the evolved amplitudes are obtained from the code here: [https://github.com/hejajama/rcbkdipole](https://github.com/hejajama/rcbkdipole). The initial conditions are the so-called MV-gamma model and are obtained from [https://arxiv.org/abs/1012.4408](https://arxiv.org/abs/1012.4408). The MV-gamma model reads
```math
  N(r^2, Y) = \frac{\sigma_0}{2}\left\{ 1- \exp \left[ \frac{\left( r^2 Q_{s0}^2 \right)^\gamma }{4} \ln \left( \frac{1}{r \Lambda_{QCD}}  + e_c e \right) \right] \right\}.
```

questions + comments can be sent to manley.329@osu.edu. 
