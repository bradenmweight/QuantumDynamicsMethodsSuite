# QuantumDynamicsMethodsSuite

The methods housed in the files "QuantumDynamics_Methods.py" and "spin-PLDM" comprise an assortment of mixed quantum classical methods for electronic and nuclear propagation in the diabatic representation. The corresponding model files can be used with any method and should simply be imported in the method-code import block as needed. The model file contains all necessary parameters for running the model dynamics.

Symmetric Quasi-Classical (SQC) Method [Miller and co-workers]

SQC.py 

  Windowing Schemes:                                 Square ("square") and Triangle ("triangle")
  State-specific Zero-point Energy (ZPE) Correction: True ("yes") or False ("no")

Spin-mapping Methods

spin-PLDM.py -- Partially Linearized Density Matrix (spin-PLDM) [Mannouch and Richardson, 2019]

spin-LSC.py  -- Partially Linearized Density Matrix (spin-PLDM) [Runeson and Richardson, 2020]

Fewest Switches Surface Hopping

  Vecloty Rescaling Schemes:                         Uniform energy-based rescaling ("energy")
  Decoherence Corrections:                           Instantaneous Decoherence Correction ("IDC")




Future To-do's:

1. Create main.py to monitor and handle all methods based upon a parameter file for ease of use.

2. Implement quasi-diabatic propagation scheme to each method to exemplify the utility in on-the-fly calculations

For any questions or concerns, please email bweight@ur.rochester.edu
