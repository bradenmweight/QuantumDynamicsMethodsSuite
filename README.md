### QuantumDynamicsMethodsSuite

The methods housed in this repository comprise an assortment of mixed quantum classical (MQC) methods for electronic and nuclear propagation. The corresponding model files can be used with any method and should simply be imported in the method's code import block as needed. The model file contains all necessary parameters for running the model dynamics.

Symmetric Quasi-Classical (SQC) Method [Miller and co-workers]

# SQC.py 

  Windowing Schemes:                                      Square ("square") and Triangle ("triangle")
  State-specific Zero-point Energy (ZPE) Correction:      True ("yes") or False ("no")

## Spin-mapping Methods

# spin-PLDM.py -- Partially Linearized Density Matrix (spin-PLDM) [Mannouch and Richardson, 2019]

# spin-LSC.py  -- Partially Linearized Density Matrix (spin-PLDM) [Runeson and Richardson, 2019]

# GDTWA.py     -- Generalized Discrete Truncated Wigner Approximation [Lang et al., J. Chem. Phys. 155, 024111 (2021); https://doi.org/10.1063/5.0054696]

# Fewest Switches Surface Hopping

  rescale_type: str -- 'energy', 'momentum'

  decoherece_type: str -- "None", "IDC", "EDC"
  
  EDC_PARAM: float -- 0.1 is optimal, Only applies for EDC decoherence
  
  AS_POP_INCREASE: int -- 0 or 1 -- 1 is better 0: will perform hop if active state population is increasing. 1: will reject hopping
  
  SWAP_COEFFS_HOP = int -- 0 or 1 -- 0 is better # 0: Never swap coefficients at hop, 1: Swap old and new active state coefficients at accepted hops



Future To-do's:

1. Implement quasi-diabatic propagation scheme to each method to exemplify the utility in on-the-fly calculations




For any questions or concerns, please email bweight@ur.rochester.edu
