# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Search the MSA profile for a better structure."""
import json
import os
import pathlib
import pickle
import random
import shutil
import sys
import time
from typing import Dict, Union

from absl import app
from absl import flags
from absl import logging
from alphafold.common import protein
from alphafold.common import residue_constants
from alphafold.common import confidence
from alphafold.data import pipeline
from alphafold.data import pipeline_multimer
from alphafold.data import templates
from alphafold.data.tools import hhsearch
from alphafold.data.tools import hmmsearch
from alphafold.model import config
from alphafold.model import data
from alphafold.model import model
from alphafold.model import modules_multimer
import jax
import jax.numpy as jnp
#from alphafold.relax import relax
import numpy as np
import glob
from collections import Counter
from scipy.special import softmax
import optax
import pandas as pd

import pdb

#JAX will preallocate 90% of currently-available GPU memory when the first JAX operation is run.
#This prevents this
import os

logging.set_verbosity(logging.INFO)

flags.DEFINE_list(
    'fasta_paths', None, 'Paths to FASTA files, each containing a prediction '
    'target that will be folded one after another. If a FASTA file contains '
    'multiple sequences, then it will be folded as a multimer. Paths should be '
    'separated by commas. All FASTA paths must have a unique basename as the '
    'basename is used to name the output directories for each prediction.')

flags.DEFINE_string('data_dir', None, 'Path to directory of supporting data.')
flags.DEFINE_string('output_dir', None, 'Path to a directory that will '
                    'store the results.')
flags.DEFINE_string('feature_dir', None, 'Path to a directory that contains the feats.')
flags.DEFINE_enum('model_preset', 'monomer',
                  ['monomer', 'monomer_casp14', 'monomer_ptm', 'multimer'],
                  'Choose preset model configuration - the monomer model, '
                  'the monomer model with extra ensembling, monomer model with '
                  'pTM head, or multimer model')
flags.DEFINE_integer('random_seed', None, 'The random seed for the data '
                     'pipeline. By default, this is randomly generated. Note '
                     'that even if this is set, Alphafold may still not be '
                     'deterministic, because processes like GPU inference are '
                     'nondeterministic.')
flags.DEFINE_integer('num_recycles', None, 'The number of recycles.')
flags.DEFINE_float('confidence_threshold', 0.9, 'What model confidence to rate successful models by.')
flags.DEFINE_integer('max_iter', None, 'The number of iterations to perform at a max.')
flags.DEFINE_float('learning_rate', 0.9, 'Learning rate for gradient updates.')

"""Predicts structure using AlphaFold for the given sequence.
This script assumes features have already been generated."""


FLAGS = flags.FLAGS


def _check_flag(flag_name: str,
                other_flag_name: str,
                should_be_set: bool):
  if should_be_set != bool(FLAGS[flag_name].value):
    verb = 'be' if should_be_set else 'not be'
    raise ValueError(f'{flag_name} must {verb} set when running with '
                     f'"--{other_flag_name}={FLAGS[other_flag_name].value}".')


##########################FUNCTIONS##########################

def get_confidence_metrics(prediction_result):
  """Post processes prediction_result to get confidence metrics."""
  confidence_metrics = {}
  confidence_metrics['plddt'] = confidence.compute_plddt(
      prediction_result['predicted_lddt']['logits'])
  if 'predicted_aligned_error' in prediction_result:
    confidence_metrics.update(confidence.compute_predicted_aligned_error(
        logits=prediction_result['predicted_aligned_error']['logits'],
        breaks=prediction_result['predicted_aligned_error']['breaks']))
    confidence_metrics['ptm'] = confidence.predicted_tm_score(
        logits=prediction_result['predicted_aligned_error']['logits'],
        breaks=prediction_result['predicted_aligned_error']['breaks'],
        asym_id=None)

    # Compute the ipTM only for the multimer model.
    confidence_metrics['iptm'] = confidence.predicted_tm_score(
          logits=prediction_result['predicted_aligned_error']['logits'],
          breaks=prediction_result['predicted_aligned_error']['breaks'],
          asym_id=prediction_result['predicted_aligned_error']['asym_id'],
          interface=True)
    confidence_metrics['ranking_confidence'] = (
          0.8 * confidence_metrics['iptm'] + 0.2 * confidence_metrics['ptm'])

  return confidence_metrics

def get_clashes(prediction_result, clash_dist=2):
    """Get clashes
    """

    final_atom_pos = prediction_result['structure_module']['final_atom_positions']
    aa_mask = prediction_result['structure_module']['final_atom_mask']
    #Get the atom pos
    atom_pos = jnp.reshape(final_atom_pos, (-1,3))
    mask = aa_mask.flatten()
    atom_pos = atom_pos*jnp.repeat(jnp.expand_dims(mask,axis=1),3,axis=1)
    #Get dists
    dists = jnp.sqrt(1e-6+jnp.sum((atom_pos[:,None]-atom_pos[None,:])**2,axis=-1))

    #dists below 2 subtracted with the zeros
    num_clashes = jnp.sum((dists < clash_dist).astype(jnp.float32))-jnp.sum((dists <0.01).astype(jnp.float32))
    clash_fraction = num_clashes/jnp.sum(mask)**2
    return clash_fraction


def predict_structure(
    fasta_path: str,
    fasta_name: str,
    output_dir_base: str,
    feature_dir: str,
    model_runners: Dict[str, model.RunModel],
    random_seed: int,
    confidence_threshold: float,
    max_iter: int,
    learning_rate: float,
    num_recycles: int,
    config):
  """Predicts structure using AlphaFold for the given sequence."""
  logging.info('Predicting %s', fasta_name)

  output_dir = os.path.join(output_dir_base, fasta_name)
  if not os.path.exists(output_dir):
    os.makedirs(output_dir)

  # Load out features from a pickled dictionary.
  features_output_path = os.path.join(feature_dir, 'features.pkl')
  feature_dict = np.load(features_output_path, allow_pickle=True)
  # Run the models.
  num_models = len(model_runners.keys())


  #Loop through all model runners
  for model_index, (model_name, model_runner) in enumerate(model_runners.items()):
    #Process feats
    model_random_seed = model_index + random_seed * num_models
    processed_feature_dict = model_runner.process_features(
        feature_dict, random_seed=model_random_seed)
    #Includes keys:
    #'aatype', 'residue_index', 'seq_length', 'msa', 'num_alignments',
    #'asym_id', 'sym_id', 'entity_id', 'deletion_matrix', 'deletion_mean', 'all_atom_mask',
    #'all_atom_positions', 'assembly_num_chains', 'entity_mask', 'cluster_bias_mask',
    #'bert_mask', 'seq_mask', 'msa_mask'

    #Add zero feats for 'num_templates', 'template_aatype', 'template_all_atom_mask', 'template_all_atom_positions'
    processed_feature_dict['num_templates'] = jnp.array(4, dtype='int32')
    processed_feature_dict['template_aatype'] = jnp.zeros((4, processed_feature_dict['seq_length']), dtype='int32')
    processed_feature_dict['template_all_atom_mask'] = jnp.zeros((4, processed_feature_dict['seq_length'], 37), dtype='int32') #Zeros here makes sure the rest doesn't matter
    processed_feature_dict['template_all_atom_positions'] = jnp.zeros((4, processed_feature_dict['seq_length'], 37, 3), dtype='float32')

    #Get the MSA shape
    msa_shape = processed_feature_dict['msa'].shape
    print('The MSA has shape:', msa_shape)
    processed_feature_dict['msa_shape'] = msa_shape

    #Setup run
    num_above_t = 5
    metrics = {'model_name':[], 'ranking_confidence': [], 'update_time':[]}

    #Init params - the msa bias
    """
    msa_feat = [
      msa_1hot, - x,y,23
      has_deletion, - x,y,1
      deletion_value, - x,y,1
      batch['cluster_profile'], - x,y,23 it is the cluster profile we want to influence as this contains information about the entire energy landscape
      deletion_mean_value, x,y,1
      ]
    """

    #Need to init zeros = no influence on the cluster profile
    msa_params = np.zeros((config.model.embeddings_and_evoformer.num_msa, msa_shape[1], 23)) #23 classes for aa, gap, mask
    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(msa_params)

    def loss_fn(msa_params, processed_feature_dict):
        """Calculate loss


        prediction_result:
        'distogram', 'experimentally_resolved', 'masked_msa', 'predicted_lddt', 'structure_module', 'plddt'
        """
        prediction_result = model_runner.predict(msa_params, processed_feature_dict) #The params for model runner are contained within self
        #Get confidence
        confidence = get_confidence_metrics(prediction_result)
        #Get loss (1/confidence)
        loss = 1/confidence['ranking_confidence']
        return loss, prediction_result

    def update(msa_params, opt_state, processed_feature_dict):
        (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(msa_params, processed_feature_dict)

        updates, new_opt_state = optimizer.update(grads, opt_state)
        new_params = optax.apply_updates(msa_params, updates)
        return loss, aux, new_params, new_opt_state

    #Define the plDDT bins
    bin_width = 1.0 / 50
    bin_centers = np.arange(start=0.5 * bin_width, stop=1.0, step=bin_width)

    #Sample
    for i in range(max_iter):
        if (np.array(metrics['ranking_confidence'])>confidence_threshold).sum()>=num_above_t:
            print('Confidence threshold reached.')
            sys.exit()


        #Predict
        t0= time.time()
        loss, aux, msa_params, opt_state = update(msa_params, opt_state, processed_feature_dict)
        t1 = time.time()
        metrics['update_time'].append(t1-t0)
        #Save the ranking
        metrics['model_name'].append(str(i+1))
        metrics['ranking_confidence'].append(1/np.mean(loss))
        metric_df = pd.DataFrame.from_dict(metrics)
        metric_df['lr'] = learning_rate
        metric_df['recycles'] = num_recycles
        metric_df.to_csv(output_dir+'/metrics_'+str(learning_rate)+'_'+str(num_recycles)+'.csv', index=None)

        #Print how its going
        print('Step:', i+1, '|confidence:', 1/np.mean(loss), '|best confidence:', np.max(metrics['ranking_confidence']))

        # Add the predicted LDDT in the b-factor column.
        plddt_per_pos = jnp.sum(jax.nn.softmax(aux['predicted_lddt']['logits']) * bin_centers[None, :], axis=-1)
        plddt_b_factors = np.repeat(plddt_per_pos[:, None], residue_constants.atom_type_num, axis=-1)
        unrelaxed_protein = protein.from_prediction(features=processed_feature_dict, result=aux,  b_factors=plddt_b_factors, remove_leading_feature_dimension=not model_runner.multimer_mode)
        unrelaxed_pdb = protein.to_pdb(unrelaxed_protein)
        unrelaxed_pdb_path = os.path.join(output_dir+'/', 'unrelaxed_'+str(i+1)+'.pdb')
        with open(unrelaxed_pdb_path, 'w') as f:
            f.write(unrelaxed_pdb)





########Here the model runners are setup and the predict function is called##########

def main(argv):
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  # Check for duplicate FASTA file names.
  fasta_names = [pathlib.Path(p).stem for p in FLAGS.fasta_paths]
  if len(fasta_names) != len(set(fasta_names)):
    raise ValueError('All FASTA paths must have a unique basename.')

  model_runners = {}
  model_names = config.MODEL_PRESETS[FLAGS.model_preset]
  for model_name in model_names[:1]: #Only use the first model here
    model_config = config.model_config(model_name)
    #Set the dropout in the str module to 0 -  from AFsample: https://github.com/bjornwallner/alphafoldv2.2.0/blob/main/run_alphafold.py
    model_config.model.heads.structure_module.dropout=0.0
    model_config.model.num_ensemble_eval = 1 #Use 1 ensemble
    model_config.model.num_ensemble_train = 1 #Use 1 ensemble
    model_config.model.num_recycle = FLAGS.num_recycles
    model_params = data.get_model_haiku_params(
        model_name=model_name, data_dir=FLAGS.data_dir)
    model_runner = model.RunModel(model_config, model_params, is_training=True) #Set training to true to have dropout in the Evoformer

    model_runners[f'{model_name}_pred_{0}'] = model_runner

  logging.info('Have %d models: %s', len(model_runners),
               list(model_runners.keys()))

  random_seed = FLAGS.random_seed #This is used in the feature processing
                                    #For multimers, this is simply returned though: if self.multimer_mode:
                                                                                    #    return raw_features
                                    #This is because the sampling happens within the multimer runscript
                                    #A new random key is fetched each iter
  if random_seed is None:
    random_seed = random.randrange(sys.maxsize // len(model_runners))
  logging.info('Using random seed %d for the data pipeline', random_seed)

  # Predict structure
  # This runs the MSA diffusion to improve the sampling efficiency
  # It looks for the previous opt state to continue where you left off if the run was interrupted
  for i, fasta_path in enumerate(FLAGS.fasta_paths):
    fasta_name = fasta_names[i]
    predict_structure(
        fasta_path=fasta_path,
        fasta_name=fasta_name,
        output_dir_base=FLAGS.output_dir,
        feature_dir=FLAGS.feature_dir,
        model_runners=model_runners,
        random_seed=random_seed,
        confidence_threshold=FLAGS.confidence_threshold,
        max_iter=FLAGS.max_iter,
        learning_rate=FLAGS.learning_rate,
        num_recycles=FLAGS.num_recycles,
        config=model_config)


if __name__ == '__main__':
  flags.mark_flags_as_required([
      'fasta_paths',
      'output_dir',
      'data_dir',
  ])

  app.run(main)
