#!/bin/bash

for j in {0..3}; do 
    for k in {0..3}; do 
        echo "Submitting $j $k"
        sbatch submit.spin-PLDM_nm $j $k # For only spin-PLDM
    done
done