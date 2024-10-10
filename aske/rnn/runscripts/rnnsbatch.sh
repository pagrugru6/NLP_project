#!/bin/bash

# normal cpu stuff: allocate cpus, memory

#SBATCH -o ../slurm_out/%j.out # STDOUT

#SBATCH --ntasks=1 --cpus-per-task=10 --mem=16000M

# we run on the gpu partition and we allocate 2 titanrtx gpus

#SBATCH -p gpu --gres=gpu:titanrtx:1

#We expect that our program should not run longer than 1 hours

#Note that a program will be killed once it exceeds this time!

#SBATCH --time=24:00:00



#your script, in this case: write the hostname and the ids of the chosen gpus.

echo ""

echo "============== RUNNING LATENT SPACE AUTOENCODER =============="

echo ""

echo $CUDA_VISIBLE_DEVICES

./vgaerun.sh

# python run.py

echo "... done <3"


