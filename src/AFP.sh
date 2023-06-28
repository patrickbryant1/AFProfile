
# Script for predicting with AlphaFold-multimer using directed sampling (=dropout, This is the procedure implemented by Wallner in CASP15: https://www.biorxiv.org/content/10.1101/2022.12.20.521205v3)
# using a diffusion process over the profile
# Fill in all the variables below to run the predictions.
# This script assumes that all python packages necessary are in the current path.

#Get ID
ID=H1144
FASTA_PATHS=../../data/casp_15/H1144/H1144.fasta
PARAMDIR=../../data/ #If v2 is used: Change the _v3 to _v2 in the multimer MODEL_PRESETS in config.py
OUTDIR=../../data/casp15/
AFDIR=./

#1. Get MSAs: run_alphafold_msa_template_only.py - this produces the feats as well (saved as pickle)
#2. Get features from MSAs: run_alphafold_feats_from_msa.py
#python3 $AFDIR/run_alphafold_feats_from_msa.py	--fasta_paths=$FASTA_PATHS \
#--output_dir $OUTDIR

#3. Learn residuals to improve the confidence: run_AFP.py

#Run AFM
MODEL_PRESET='multimer'
NUM_RECYCLES=0
CONFIDENCE_T=0.99
MAX_ITER=500
LR=0.0001

mkdir $OUTDIR/$ID'/lr_'$LR'_nr_'$NUM_RECYCLES

#Run
python3 $AFDIR/run_AFP.py --fasta_paths=$FASTA_PATHS \
--data_dir=$PARAMDIR --model_preset=$MODEL_PRESET \
--num_recycles=$NUM_RECYCLES \
--confidence_threshold=$CONFIDENCE_T \
--max_iter=$MAX_ITER \
--learning_rate=$LR \
--output_dir=$OUTDIR
