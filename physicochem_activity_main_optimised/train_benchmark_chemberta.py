#!/usr/bin/env python3


import os
import sys
import pickle
import numpy as np
import pandas as pd
from loguru import logger

# Disable TF warnings
os.environ["TF_USE_LEGACY_KERAS"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = ''

from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Logging
logger.remove()
logger.add(sys.stderr, level="INFO")
logger.add(
    "benchmark_chemberta_simple_{time}.txt",
    rotation="500 MB", retention="10 days", compression="zip", level="DEBUG"
)

print("=" * 80)
print("BENCHMARK: Simple ChemBERTa (Ligand) + Mutation-Class One-Hot")
print("No dropout, no custom components.")
print("=" * 80)

# =============================================================================
# DATA LOADING
# =============================================================================
print("\nLoading datasets...")
script_dir = os.path.dirname(os.path.abspath(__file__))
df_train   = pd.read_csv(os.path.join(script_dir, 'validated_july_2026_trainset_valid_n_nonvalid_tki.csv'))

df_train.columns = df_train.columns.str.strip()

# Dataset fix: normalize whitespace in 'tkd' (one row is ' l858r/t790m/c797s triple' with a
# stray leading space, which would otherwise be treated as a spurious 9th mutation class).
df_train['tkd'] = df_train['tkd'].astype(str).str.strip()

ligand_smiles    = df_train['smiles']
mutant           = df_train['tkd']
activity_values  = df_train['standard value']
docking_values   = df_train['dock']



print(f"Training samples : {len(ligand_smiles)}")


# Valid sample filter
valid_mask = ~(
    ligand_smiles.isna() |
    mutant.isna() |
    activity_values.isna() |
    docking_values.isna()
)
valid_sample_count = valid_mask.sum()
print(f"\n✓ Valid samples: {valid_sample_count}/{len(df_train)}")
if valid_sample_count == 0:
    sys.exit(1)

df_train_valid         = df_train[valid_mask].copy().reset_index(drop=True)
ligand_smiles_valid    = df_train_valid['smiles']
mutant_valid           = df_train_valid['tkd']
activity_values_valid  = df_train_valid['standard value']
docking_values_valid   = df_train_valid['dock'].values

# =============================================================================
# CHEMBERTA UTILITIES
# =============================================================================
def get_device():
    try:
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    except Exception:
        return torch.device('cpu')

def load_chemberta(model_name: str = 'seyonec/ChemBERTa-zinc-base-v1', device=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    if device is None:
        device = get_device()
    model.to(device)
    model.eval()
    logger.info(f"ChemBERTa loaded: {model_name}  |  device: {device}")
    return tokenizer, model, device

def get_chemberta_embeddings(smiles_list, tokenizer, model, device, batch_size=64):
    inputs  = tokenizer(smiles_list, return_tensors='pt',
                        padding=True, truncation=True)
    dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'])
    loader  = DataLoader(dataset, batch_size=batch_size)
    embs    = []
    with torch.no_grad():
        for ids, mask in loader:
            ids, mask = ids.to(device), mask.to(device)
            out    = model(input_ids=ids, attention_mask=mask, return_dict=True)
            pooled = getattr(out, 'pooler_output', None)
            if pooled is None:
                last   = out.last_hidden_state
                maskf  = mask.unsqueeze(-1).float()
                pooled = (last * maskf).sum(1) / maskf.sum(1).clamp(min=1e-9)
            embs.append(pooled.cpu())
    return torch.cat(embs, dim=0).numpy()

# =============================================================================
# CATEGORICAL PROTEIN ENCODING
# =============================================================================
def fit_mutation_encoder(mutation_series):
    le = LabelEncoder()
    le.fit(mutation_series.dropna().astype(str))
    logger.info(f"Mutation classes ({len(le.classes_)}): {list(le.classes_)}")
    return le, len(le.classes_)

def encode_mutation_onehot(mutation_series, le):
    n_classes = len(le.classes_)
    labels    = mutation_series.astype(str).values
    indices   = []
    for lbl in labels:
        if lbl in le.classes_:
            indices.append(le.transform([lbl])[0])
        else:
            indices.append(-1)
    onehot = np.zeros((len(labels), n_classes), dtype=np.float32)
    for row_i, idx in enumerate(indices):
        if idx >= 0:
            onehot[row_i, idx] = 1.0
    return onehot

# =============================================================================
# SIMPLE MLP MODEL DEFINITION
# =============================================================================
def build_simple_benchmark_model(ligand_dim=768, mutation_dim=8):
    """
    Pure dense MLP taking ChemBERTa embeddings and one-hot mutation encoded array.
    No dropout, no custom mechanics layers.
    """
    lig_input = Input(shape=(ligand_dim,), name='ligand_embedding')
    mut_input = Input(shape=(mutation_dim,), name='mutation_onehot')

    # Simple concatenation
    x = Concatenate(name='concat_inputs')([lig_input, mut_input])

    # Plain Dense standard layers (No dropout, No BN)
    x = Dense(256, activation='relu', kernel_initializer='he_normal', name='dense_1')(x)
    x = Dense(64, activation='relu', kernel_initializer='he_normal', name='dense_2')(x)

    # Output Heads
    activity_output = Dense(1, activation='linear', 
                            kernel_initializer='glorot_uniform', 
                            name='activity_output')(x)
                            
    docking_output  = Dense(1, activation='linear', 
                            kernel_initializer='glorot_uniform', 
                            name='docking_output')(x)

    model = Model(
        inputs=[lig_input, mut_input],
        outputs=[activity_output, docking_output],
        name='simple_chemberta_benchmark'
    )
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={'activity_output': 'mean_squared_error',
              'docking_output':  'mean_squared_error'},
        loss_weights={'activity_output': 1.0, 'docking_output': 0.6},
        metrics={'activity_output': ['mae', 'mse'],
                 'docking_output':  ['mae', 'mse']}
    )
    model.summary(print_fn=logger.info)
    return model

# =============================================================================
# MAIN EXECUTION
# =============================================================================
def main():
    device = get_device()
    print('\nLoading ChemBERTa model (ligand encoder only)...')
    tokenizer, chem_model, device = load_chemberta(device=device)

    print('\nComputing ChemBERTa embeddings for training ligands...')
    lig_smiles_list = ligand_smiles_valid.astype(str).tolist()
    lig_embs_raw    = get_chemberta_embeddings(lig_smiles_list, tokenizer, chem_model, device)

    # Scale ChemBERTa embeddings
    chem_emb_scaler = StandardScaler()
    lig_embs = chem_emb_scaler.fit_transform(lig_embs_raw)

    print('\nEncoding protein as mutation-class one-hot...')
    le, n_mutation_classes = fit_mutation_encoder(mutant_valid)
    mut_onehot_all = encode_mutation_onehot(mutant_valid, le)
    print(f"Mutation one-hot shape  : {mut_onehot_all.shape} (classes: {list(le.classes_)})")

    # Scalers for Target values
    y_train_act = np.log1p(activity_values_valid.values)
    y_scaler_act = StandardScaler()
    y_train_act_scaled = y_scaler_act.fit_transform(y_train_act.reshape(-1, 1)).flatten()

    y_train_dock = docking_values_valid
    y_scaler_dock = StandardScaler()
    y_train_dock_scaled = y_scaler_dock.fit_transform(y_train_dock.reshape(-1, 1)).flatten()

    # Build and train Simple Model
    print("\n" + "=" * 80)
    print("TRAINING SIMPLE MLP BENCHMARK MODEL")
    print("=" * 80)

    model = build_simple_benchmark_model(ligand_dim=768, mutation_dim=n_mutation_classes)
    
    checkpoint = ModelCheckpoint(
        'chemberta_simple_model.h5',
        monitor='val_loss', save_best_only=True, verbose=1)
        
    early_stop = EarlyStopping(
        monitor='val_loss', patience=40,
        restore_best_weights=True, verbose=1)

    history = model.fit(
        x=[lig_embs, mut_onehot_all],
        y={'activity_output': y_train_act_scaled,
           'docking_output':  y_train_dock_scaled},
        epochs=150,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop, checkpoint],
        verbose=1,
    )

    # Save scalers and label encoder
    with open('benchmark_simple_scalers.pkl', 'wb') as f:
        pickle.dump({
            'y_scaler_act': y_scaler_act, 
            'y_scaler_dock': y_scaler_dock,
            'chem_emb_scaler': chem_emb_scaler
        }, f)
        
    with open('benchmark_simple_le.pkl', 'wb') as f:
        pickle.dump(le, f)

    print('\nModel, scalers, and label encoder saved successfully.')

   
    print("\n✓ Simple Benchmark Training Script completed.")

if __name__ == "__main__":
    main()
