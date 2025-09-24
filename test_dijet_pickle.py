import pickle
import dijet

d = dijet.DIJET(fit_type='pp', constrained_moments=True, IR_reg=['gauss', 0.5], nucleon='p')
pickle.dumps(d)   # will this succeed?