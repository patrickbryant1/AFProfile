# AFProfile
Improved protein complex prediction with AlphaFold-multimer by denoising the MSA profile.
\
AFProfile manipulates the *MSA representation* by learning residuals to the MSA profile that are *more useful for the network*.
This is analogous to a denoising process over the MSA and proves to be a highly efficient process resulting in *more accurate structures*.
The process can be seen as denoising the MSA representation, similar to how a *blurry image would be sharpened*.


\
<img src="./AFP.svg"/>
\
\
AlphaFold2 (including AlphaFold-multimer) is available under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0) and so is AFProfile, which is a derivative thereof.  \
The AlphaFold2 parameters are made available under the terms of the [CC BY 4.0 license](https://creativecommons.org/licenses/by/4.0/legalcode) and have not been modified.
\
**You may not use these files except in compliance with the licenses.**

## Optimisation for 6nnw
- Here is an example trajectory for PDBID 6nnw sorted by confidence.

<img src="./6nnw.gif"/>

- The final prediction has an MMscore of 0.96 compared to 0.44 using AF-multimer. The [native structure](https://www.rcsb.org/structure/6NNW) is in grey.

<img src="./6nnw.svg"/>

The confidence used to denoise the MSA is defined as: \
Confidence = 0.8 iptm + 0.2 ptm \
Where iptm is the predicted TM-score in the interface and ptm that of the entire complex.

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

## Install the AlphaFold requirements

- For the python environment, we recommend to install it with pip as described below. You can do this in your virtual environment of choice.
- Otherwise, you can follow the installation with docker here: https://github.com/deepmind/alphafold/tree/main
```
pip install -U jaxlib==0.3.24+cuda11.cudnn82 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
pip install jax==0.3.24
pip install ml-collections==0.1.1
pip install dm-haiku==0.0.9
pip install pandas==1.3.5
pip install biopython==1.81
pip install chex==0.1.5
pip install dm-tree==0.1.8
pip install immutabledict==2.0.0
pip install numpy==1.21.6
pip install scipy==1.7.3
pip install tensorflow-cpu==2.12.0
pip install tensorflow==2.11.0
pip install optax==0.1.4
```

## Try the test case
Now when you have installed the required packages - you can run a test case on CASP15 target H1144
```
cd src
bash AFP.sh
```

## Install the genetic search programs
- We install the genetic search programs from source. This will make the searches faster.

*hh-suite*
```
cd src
git clone https://github.com/soedinglab/hh-suite.git
mkdir -p hh-suite/build && cd hh-suite/build
cmake -DCMAKE_INSTALL_PREFIX=. ..
make -j 4 && make install
cd ..
```

*hmmer*
```
cd src
wget http://eddylab.org/software/hmmer/hmmer.tar.gz
tar -xvzf hmmer.tar.gz
rm hmmer.tar.gz
cd hmmer-3.3.2
./configure
make
cd ..
```

*kalign*
```
wget https://github.com/TimoLassmann/kalign/archive/refs/tags/v3.3.2.tar.gz
tar -zxvf v3.3.2.tar.gz
rm v3.3.2.tar.gz
cd kalign-3.3.2/
./autogen.sh
bash configure
make
make check
make install
cd ..
```


## Download all databases for AlphaFold
- If you have already installed AlphaFold, you don't need to do this. Then you can simply
provide the paths for the databases in the runscript.

*Small BFD: 17 GB*
```
wget https://storage.googleapis.com/alphafold-databases/reduced_dbs/bfd-first_non_consensus_sequences.fasta.gz
gunzip bfd-first_non_consensus_sequences.fasta.gz
mkdir data/small_bfd
mv bfd-first_non_consensus_sequences.fasta data/small_bfd
rm bfd-first_non_consensus_sequences.fasta.gz
```

*UNIREF90: 67 GB*
```
wget https://ftp.ebi.ac.uk/pub/databases/uniprot/uniref/uniref90/uniref90.fasta.gz
gunzip uniref90.fasta.gz
mkdir data/uniref90
mv uniref90.fasta data/uniref90/
rm uniref90.fasta.gz
```

*UNIPROT: 105 GB*
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

- The following template databases are not used for the predictions, but needed due to the feature processing.

*PDB SEQRES: 0.2 GB*
```
wget https://ftp.wwpdb.org/pub/pdb/derived_data/pdb_seqres.txt
mkdir pdb_seqres
mv pdb_seqres.txt  pdb_seqres/
```

*MGNIFY: 120 GB*
```
wget https://storage.googleapis.com/alphafold-databases/v2.3/mgy_clusters_2022_05.fa.gz
gunzip mgy_clusters_2022_05.fa.gz
mkdir mgnify
mv mgy_clusters_2022_05.fa.gz mgnify/
rm mgy_clusters_2022_05.fa.gz
```

*MMCIF: 238 GB*
- This may take a while...
```
mkdir -p data/pdb_mmcif/raw
mkdir data/pdb_mmcif/mmcif_files
rsync --recursive --links --perms --times --compress --info=progress2 --delete --port=33444 rsync.rcsb.org::ftp_data/structures/divided/mmCIF/ data/pdb_mmcif/raw

find data/pdb_mmcif/raw -type f -iname "*.gz" -exec gunzip
find data/pdb_mmcif/raw -type d -empty -delete  
for subdir in data/pdb_mmcif/raw/*
do
  mv "${subdir}/"*.cif data/pdb_mmcif/mmcif_files/
done
find data/pdb_mmcif/raw -type d -empty -delete
```
