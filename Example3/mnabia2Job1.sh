#!/bin/bash  
#SBATCH --job-name="mnabia2Job1"  
#SBATCH --output="Main.%j.%N.out"  
#SBATCH --partition=compute  
#SBATCH --nodes=1  
#SBATCH --ntasks-per-node=24 
#SBATCH --export=ALL  
#SBATCH -t 01:00:00   

python Main.py
