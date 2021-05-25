# QuantumDynamicsMethodsSuite

The methods housed in the files "QuantumDynamics_Methods.py" and "spin-PLDM" comprise an assortment of mixed quantum classical methods for electronic and nuclear propagation in the diabatic representation. The corresponding model files can be used with any method and should simply be imported in the method-code import block as needed. The model file contains all necessary parameters for running the model dynamics.

QuantumDynamics_Methods.py contains the following methods: 

1. PLDM (Huo and Coker) 

2. SQC with all variations of windowing and gamma-corrections -- square and triangle windows; gamma-fixed and -adjusted (William Miller and co-workers)

spin-PLDM.py

1. Partially linearized spin-mapping method (Mannouch and Richardson) 

Future To-do's:

1. Create main.py to monitor and handle all methods based upon a parameter file for ease of use.

2. Implement quasi-diabatic propagation scheme to each method to exemplify the utility in on-the-fly calculations (Mandal and Huo)

For any questions or concerns, please send an email to bweight@ur.rochester.edu
