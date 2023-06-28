# AFProfile
Improved protein complex prediction with AlphaFold-multimer by denoising the MSA profile.
\
AFProfile manipulates the MSA representation by learning residuals to the MSA profile that are more useful for the network.
This is analogous to a denoising diffusion process over the MSA and proves to be a highly efficient process resulting in better structures.

\
<img src="./AFP.svg"/>
\
\
AlphaFold2 (including AlphaFold-multimer) is available under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0) and so is AFProfile, which is a derivative thereof.  \
The AlphaFold2 parameters are made available under the terms of the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/legalcode) and have not been modified.
\
**You may not use these files except in compliance with the licenses.**



# Setup

## Get the AlphaFold-multimer parameters
```
mkdir data/params
wget https://storage.googleapis.com/alphafold/alphafold_params_2022-03-02.tar
tar -xf alphafold_params_2022-03-02.tar
mv params_model_1.npz data/params
rm *.npz
rm alphafold_params_2022-03-02.tar
```
## Download all databases for AlphaFold
- If you have already installed AlphaFold, you don't need to do this. Then you can simply
provide the paths for the databases in the runscript.

*Small BFD*
```
wget https://storage.googleapis.com/alphafold-databases/reduced_dbs/bfd-first_non_consensus_sequences.fasta.gz
gunzip bfd-first_non_consensus_sequences.fasta.gz
mkdir data/small_bfd
mv bfd-first_non_consensus_sequences.fasta data/small_bfd
rm bfd-first_non_consensus_sequences.fasta.gz
```

*UNIREF90*
```
wget https://ftp.ebi.ac.uk/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz
gunzip uniref90.fasta.gz
mkdir data/uniref90
mv uniref90.fasta data/uniref90/
rm uniref90.fasta.gz

```
*UNIREF30*
```
wget https://storage.googleapis.com/alphafold-databases/v2.3/UniRef30_2021_03.tar.gz
tar -xvzf UniRef30_2021_03.tar.gz
mkdir data/uniref30
mvUniRef30_2021_03* data/uniref30/
```

*UNIPROT*
```
wget https://ftp.ebi.ac.uk/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_trembl.fasta.gz
wget https://ftp.ebi.ac.uk/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz
gunzip uniprot_trembl.fasta.gz
gunzip uniprot_sprot.fasta.gz
mkdir data/uniprot
cat uniprot_sprot.fasta >> uniprot_trembl.fasta
mv uniprot_trembl.fasta data/uniprot/uniprot.fasta
rm *.gz
rm uniprot_sprot.fasta
```

MGNIFY=$DATADIR/mgnify/mgy_clusters_2022_05.fa
PDB70=$DATADIR/pdb70_from_mmcif_220313
PDBSEQRES=$DATADIR/pdb_seqres/pdb_seqres.txt
MMCIFDIR=$DATADIR/pdb_mmcif/

## Install the
