#!/bin/bash -l
#$ -P ivc-ml
#$ -pe omp 4
#$ -l h_rt=2:00:00
#$ -N ldm_setup_env
#$ -j y
#$ -o /projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE/ldm_official/setup_logs/setup_env.qlog
bash /projectnb/ivc-ml/mvasu6/Perceptually-Better-VAE/ldm_official/setup_env.sh
