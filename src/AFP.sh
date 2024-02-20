
# Script for predicting with AlphaFold-multimer using directed sampling (=dropout, This is the procedure implemented by Wallner in CASP15: https://www.biorxiv.org/content/10.1101/2022.12.20.521205v3)
# using a denoising process over the profile. The MSA is denoised by doing gradient descent through AlphaFold-multimer.
# Fill in all the variables below to run the predictions.
# This script assumes that all python packages necessary are in the current path.
#The fasta conventions are the same as for AlphaFold-multimer. See the example files in ../data/

#Get ID
ID=T1123
FASTA_PATHS=../data/T1123/T1123.fasta
PARAMDIR=../data/ #If v2 is used: Change the _v3 to _v2 in the multimer MODEL_PRESETS in config.py
OUTDIR=../data/
AFDIR=./

#1. Get MSAs: run generate_msas.sh which runs: run_alphafold_msa_template_only.py - this produces the feats as well (saved as pickle)
#For this test case - the features have already been generated and are available here: ../data/T1123/features.pkl

#2. Learn residuals to improve the confidence: run_AFP.py
#Run AFM
MODEL_PRESET='multimer'
NUM_RECYCLES=20 #Number of recycles
CONFIDENCE_T=0.95 #At what confidence to stop the search
MAX_ITER=500 #Max number of iterations
LR=0.0001 #Learning rate for ADAM optimizer

#Run
python3 $AFDIR/run_AFP.py --fasta_paths=$FASTA_PATHS \
--data_dir=$PARAMDIR --model_preset=$MODEL_PRESET \
--num_recycles=$NUM_RECYCLES \
--confidence_threshold=$CONFIDENCE_T \
--max_iter=$MAX_ITER \
--learning_rate=$LR \
--output_dir=$OUTDIR \
--feature_dir=$OUTDIR/$ID/ 
