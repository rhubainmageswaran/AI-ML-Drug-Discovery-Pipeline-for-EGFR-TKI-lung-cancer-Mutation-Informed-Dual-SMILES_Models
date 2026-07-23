import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, MolSurf, GraphDescriptors, Fragments
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit import DataStructs
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
from numpy.linalg import norm
from loguru import logger
import logging
import sys

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate, Embedding, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from rdkit import RDLogger

import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

# === Configuration ===
RDLogger.DisableLog('rdApp.*')
np.random.seed(42)

# Logger configuration
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)
logger.add(
    "adv_physchem5f_{time}.txt",
    rotation="500 MB",
    retention="10 days",
    compression="zip",
    level="DEBUG"
)

print("="*80)
print("Simple Neural Network with Advanced Physicochemical Descriptors")
print("Feed-Forward Neural Network Model")
print("="*80)

# === Load Data ===
print("\nLoading datasets...")
script_dir = os.path.dirname(os.path.abspath(__file__))
df_train = pd.read_csv(os.path.join(script_dir, 'validated_july_2026_trainset_valid_n_nonvalid_tki.csv'))


df_train.columns = df_train.columns.str.strip()

# Dataset fix: 'smiles_sequence_hinge_loop_t790m_c797s' is saved twice (byte-identical
# duplicate column). Pandas auto-renames the 2nd occurrence to '...t790m_c797s.1'; drop it.
if 'smiles_sequence_hinge_loop_t790m_c797s.1' in df_train.columns:
    df_train = df_train.drop(columns=['smiles_sequence_hinge_loop_t790m_c797s.1'])

# Dataset fix: normalize whitespace in 'tkd' (one row is ' l858r/t790m/c797s triple' with a
# stray leading space, which would otherwise be treated as a spurious 9th mutation class).
df_train['tkd'] = df_train['tkd'].astype(str).str.strip()

#smiles for feature capture of ligand and mutation protein substructures
#Mechanistic reaction order for EGFR: full -> ATP pocket -> P-loop -> C-helix ->
#19-deletions -> hinge loop -> A-loop DFG -> HRD catalytic motif
ligand_smiles = df_train['smiles']
full_smiles = df_train['smiles_full_sequence_egfr_manual']
mutation_smiles = df_train['smiles_sequence_atp_ pocket']
mut_p_loop = df_train['smiles_sequence_p_loop_constant']
mut_helix = df_train['smiles_sequence_c_helix_constant']
mut_19_del = df_train['smiles_sequence_19_deletions']
mut_hinge_loop = df_train['smiles_sequence_hinge_loop_t790m_c797s']
mut_dfg_a_loop = df_train['smiles_sequence_a_loop_dfg']
mut_hrd_cat = df_train['smiles_sequence_hrd_constant']

mutant = df_train['tkd'] #label for mutation type

activity_values = df_train['standard value'] #y_train1 target
docking_values = df_train['dock'] #y_train2 target



print(f"Training samples: {len(ligand_smiles)}")


# Data Validation
print("\n" + "="*80)
print("DATA VALIDATION")
print("="*80)

mutation_site_columns = {
    'FULL_SMILES': full_smiles,
    'ATP_POCKET': mutation_smiles,
    'P_LOOP': mut_p_loop,
    'C_HELIX': mut_helix,
    'DEL19': mut_19_del,
    'HINGE_LOOP': mut_hinge_loop,
    'DFG_A_LOOP': mut_dfg_a_loop,
    'HRD_CAT': mut_hrd_cat
}

for site_name, site_series in mutation_site_columns.items():
    missing_count = site_series.isna().sum()
    print(f"  {site_name:15s}: {missing_count:4d} missing SMILES ({missing_count/len(site_series)*100:.2f}%)")

# Filter to valid samples
valid_mask = ~(
    ligand_smiles.isna() | 
    full_smiles.isna() |
    mutation_smiles.isna() | 
    mut_p_loop.isna() | 
    mut_helix.isna() | 
    mut_19_del.isna() |
    mut_hinge_loop.isna() |
    mut_dfg_a_loop.isna() | 
    mut_hrd_cat.isna() | 
    activity_values.isna() |
    docking_values.isna() 
)

valid_sample_count = valid_mask.sum()
print(f"\n✓ Valid samples: {valid_sample_count}/{len(df_train)} ({valid_sample_count/len(df_train)*100:.2f}%)")

if valid_sample_count == 0:
    print("\n❌ ERROR: No complete samples found!")
    sys.exit(1)

df_train_valid = df_train[valid_mask].copy().reset_index(drop=True)


ligand_smiles_valid = df_train_valid['smiles']
full_smiles_valid = df_train_valid['smiles_full_sequence_egfr_manual']
mutation_smiles_valid = df_train_valid['smiles_sequence_atp_ pocket']
mut_p_loop_valid = df_train_valid['smiles_sequence_p_loop_constant']
mut_helix_valid = df_train_valid['smiles_sequence_c_helix_constant']
mut_19_del_valid = df_train_valid['smiles_sequence_19_deletions']
mut_hinge_loop_valid = df_train_valid['smiles_sequence_hinge_loop_t790m_c797s']
mut_dfg_a_loop_valid = df_train_valid['smiles_sequence_a_loop_dfg']
mut_hrd_cat_valid = df_train_valid['smiles_sequence_hrd_constant']
mutant_valid = df_train_valid['tkd']

activity_values_valid = df_train_valid['standard value']
activity_values2_valid = df_train_valid['dock'].values

# Create unique mutation profiles
# Mechanistic reaction order for EGFR: full -> ATP pocket -> P-loop -> C-helix ->
# 19-deletions -> hinge loop -> A-loop DFG -> HRD catalytic motif
mutation_profile_columns = [
    'smiles_full_sequence_egfr_manual',
    'smiles_sequence_atp_ pocket',
    'smiles_sequence_p_loop_constant',
    'smiles_sequence_c_helix_constant',
    'smiles_sequence_19_deletions',
    'smiles_sequence_hinge_loop_t790m_c797s',
    'smiles_sequence_a_loop_dfg',
    'smiles_sequence_hrd_constant',
    'tkd'
]

unique_mutation_profiles = df_train_valid[mutation_profile_columns].drop_duplicates(subset=['tkd']).reset_index(drop=True)
print(f"Unique mutation profiles: {len(unique_mutation_profiles)}")

#Division for custom interaction features using intermolecular and intramolecular forces
def safe_divide(numerator, denominator, default=0.0):
    """Safe division with default value for zero denominator"""
    if isinstance(denominator, (int, float)):
        return numerator / denominator if denominator != 0 else default
    else:
        result = np.where(denominator != 0, numerator / denominator, default)
        return result



def generate_lig_inter_features(smiles): #Intermolecular Ligand, input smiles ligand, returns np array
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    features = []
    
    try:
        #Hydrogen Bonding,
        features.append(Lipinski.NumHDonors(mol))
        features.append(Lipinski.NumHAcceptors(mol))
        features.append(Lipinski.NHOHCount(mol))
        features.append(Lipinski.NOCount(mol))
        features.append(rdMolDescriptors.CalcNumHBD(mol)) #includes N, O, and S (manual edit)
        features.append(rdMolDescriptors.CalcNumHBA(mol)) #includes N, O, and S (manual edit)
        
        #Electrostatic bonding
        #Partial charge = the small positive or negative charge assigned to each atom due to unequal sharing of electrons in bonds (like in polar bond
        features.append(Descriptors.MaxPartialCharge(mol)) #highest partial charge among all atoms in the molecule (most positive atom), (likely electrophilic)
        features.append(Descriptors.MinPartialCharge(mol)) #highest partial charge among all atoms in the molecule (most negative atom), (likely nucleophilic)
        features.append(Descriptors.MaxAbsPartialCharge(mol)) #largest absolute value of partial charge among all atom, strongest charge polarization within the molecule
        features.append(Descriptors.MaxPartialCharge(mol) - Descriptors.MinPartialCharge(mol)) #difference between most positive and most negative atoms), larger values mean stronger polarity within the molecule.
        features.append(Descriptors.MinAbsPartialCharge(mol))
        
        #Polar surface
        features.append(MolSurf.TPSA(mol))#The sum of the surface areas of all polar atoms (mostly oxygen and nitrogen) and their attached hydrogens.High TPSA -> more polar, less membrane permeable, more soluble, Low TPSA -> less polar, more membrane permeable (good for oral drugs)
        
        features.append(MolSurf.LabuteASA(mol))#An approximation of the total solvent-accessible surface area (SASA) of the molecule, calculated using Labute's algorithm. #Reflects molecular size and hydrophobic surface exposure
        #Ai​=Si​⋅Pi​, S = 4πr^2i , Pi​=1−j∑​(1−fij
        # S=total spherical surface area of atom , Pi=atomic solvation parameter, fij=fraction of atom i's surface area in contact with atom j

        features.append(Crippen.MolMR(mol))#The Ghose-Crippen formula is an atom-contribution method used to estimate the octanol-water partition coefficient (log P) and molar refractivity (MR) of a molecule.
        #Molar refractivity, a measure of the polarizability of the molecule
        
        #Size & Rigidity  (#May overlap with others)
        features.append(Descriptors.MolWt(mol)) #molecular weight
        features.append(Lipinski.HeavyAtomCount(mol))
        features.append(rdMolDescriptors.CalcNumRotatableBonds(mol))
        
        features.append(Crippen.MolLogP(mol))
        features.append(Descriptors.FractionCSP3(mol)) # fraction of SP3 hybridised carbons
        features.append(Lipinski.NumAromaticRings(mol))
        aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        features.append(aromatic_atoms)
        
        # Pi-Pi stacking 
        features.append(Descriptors.NumAromaticCarbocycles(mol))
        features.append(Descriptors.NumAromaticHeterocycles(mol))

        #Halogen
        features.append(Fragments.fr_halogen(mol))

        #Flexibility
        features.append(Lipinski.NumRotatableBonds(mol))

        return np.array(features)
        
    except Exception as e:
        print(f"Error in lig_inter: {str(e)}")
        return None


def generate_mut_inter_features(smiles): #Intermolecular Mutation, input smiles mutation, returns np array
    return generate_lig_inter_features(smiles)


#Subsequent priority of descriptors to capture features from intramolecular forces 
def generate_lig_intra_features(smiles): #Intramolecular Ligand
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    
    features = []
    
    try:
        #Covalent bond
        num_bonds = mol.GetNumBonds()
        features.append(num_bonds)
        
        #higher order bonds favours intramolecular forces within molecule, bond order indicates strength of bond
        single_bonds = sum(1 for bond in mol.GetBonds() if bond.GetBondTypeAsDouble() == 1.0)
        double_bonds = sum(1 for bond in mol.GetBonds() if bond.GetBondTypeAsDouble() == 2.0)
        triple_bonds = sum(1 for bond in mol.GetBonds() if bond.GetBondTypeAsDouble() == 3.0)
        aromatic_bonds = sum(1 for bond in mol.GetBonds() if bond.GetIsAromatic())
        features.extend([single_bonds, double_bonds, triple_bonds, aromatic_bonds])
        
        avg_bond_order = np.mean([bond.GetBondTypeAsDouble() for bond in mol.GetBonds()]) if num_bonds > 0 else 0
        features.append(avg_bond_order)
        
        #Rigidity (May Overlap), flexibility indicates less intramolecular forces within molecule
        features.append(Lipinski.NumRotatableBonds(mol))
        features.append(Lipinski.RingCount(mol))
        features.append(Lipinski.NumAromaticRings(mol))
        
        rigid_bonds = sum(1 for bond in mol.GetBonds() if bond.IsInRing())
        fraction_rigid = rigid_bonds / num_bonds if num_bonds > 0 else 0
        features.append(fraction_rigid)
        
        # Pi-Pi bonding 
        features.append(Descriptors.NumAromaticCarbocycles(mol))
        features.append(Descriptors.NumAromaticHeterocycles(mol))
        
        #Hybridization (May Overlap), branching indicates less intramolecular forces within molecule
        sp2_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetHybridization() == Chem.HybridizationType.SP2)
        sp3_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetHybridization() == Chem.HybridizationType.SP3)
        sp_carbons = sum(1 for atom in mol.GetAtoms() if atom.GetHybridization() == Chem.HybridizationType.SP)
        features.extend([sp_carbons, sp2_carbons, sp3_carbons])
        
        #Ring strain
        ring_sizes = [len(ring) for ring in mol.GetRingInfo().AtomRings()]
        avg_ring_size = np.mean(ring_sizes) if ring_sizes else 0
        min_ring_size = min(ring_sizes) if ring_sizes else 0
        features.extend([avg_ring_size, min_ring_size])
        
        three_member_rings = sum(1 for size in ring_sizes if size == 3)
        four_member_rings = sum(1 for size in ring_sizes if size == 4)
        features.extend([three_member_rings, four_member_rings])
        
        #Complexity
        features.append(GraphDescriptors.BertzCT(mol))
        features.append(GraphDescriptors.Kappa1(mol))
        features.append(GraphDescriptors.Kappa2(mol))
        features.append(GraphDescriptors.Kappa3(mol))
        features.append(rdMolDescriptors.CalcNumBridgeheadAtoms(mol))
        features.append(rdMolDescriptors.CalcNumSpiroAtoms(mol))
        
        return np.array(features)
        
    except Exception as e:
        print(f"Error in lig_intra: {str(e)}")
        return None


def generate_mut_intra_features(smiles): # Intramolecular Mutation
    return generate_lig_intra_features(smiles)

def calculate_similarity_metrics(vec1, vec2): #input np arrays, returns a dict with math metrics
    # 1. Calculate Cosine Similarity with safety check
    norm1 = norm(vec1)
    norm2 = norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        # If either vector has zero norm, return default values
        return {
            'cosine_similarity': 0.0,
            'sine_dissimilarity': 0.0,
            'dot_product': 0.0
        }
    
    cosine_sim = np.dot(vec1, vec2) / (norm1 * norm2)
    sine_of_angle = np.sqrt(1 - cosine_sim**2)

    #logger.info(f"Cosine Similarity: {cosine_sim}, Sine dssimilarity: {sine_of_angle}, Dot Product: {np.dot(vec1, vec2)}")
    
    return {
        'cosine_similarity': cosine_sim,
        'sine_dissimilarity': sine_of_angle,
        'dot_product': np.dot(vec1, vec2)
    }


def calculate_fp_metrics(smiles1, smiles2): #input smiles, returns dict with rdkit datastructs similarity
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    if mol1 is None or mol2 is None:
        return {'dice_sim': 0.0, 'tanimoto': 0.0}
    
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
    
    dice_sim = DataStructs.DiceSimilarity(fp1, fp2)
    tanimoto = DataStructs.TanimotoSimilarity(fp1, fp2)

    #logger.info(f"Dice Similarity: {dice_sim}, Tanimoto Similarity: {tanimoto}")

    
    return {
        'dice_sim': dice_sim,
        'tanimoto': tanimoto,
    }


def generate_inter_interaction_features(lig_inter, mut_inter): #similarity on intermolecular interactions ligand and mutation
    features = []
    metrics = calculate_similarity_metrics(lig_inter, mut_inter)
    
    features.append(metrics['cosine_similarity'])
    features.append(metrics['sine_dissimilarity'])
    
    return np.array(features)


def generate_intra_interaction_features(lig_intra, mut_intra): #similarity on intramolecular interactions ligand and mutation
    features = []
    metrics = calculate_similarity_metrics(lig_intra, mut_intra)
    
    features.append(metrics['cosine_similarity'])
    features.append(metrics['sine_dissimilarity'])
    
    return np.array(features)



def generate_hierarchical_features(ligand_smiles_series, mutant_id_series):
    """
    Generate features from ligand SMILES only.
    mutant_id_series should be numpy array of integers (0 to n_mutation_classes-1
    representing the 8 tkd mutation classes: wild adeno lung, del, del/t790m double,
    del/t790m/c797s triple, ins 20, subs l858r, l858r/t790m double, l858r/t790m/c797s triple).
    """
    
    lig_inter_list = []
    lig_intra_list = []
    mutant_id_list = []
    valid_indices = []
    
    print("\nGenerating hierarchical features (Ligand SMILES + Mutant ID)...")

    ligand_cache = {}
    
    for idx, (lig_smi, mut_id) in enumerate(zip(ligand_smiles_series, mutant_id_series)):
        if idx % 50 == 0:
            print(f"  Processing sample {idx}/{len(ligand_smiles_series)}...")
        
        if idx == 0:  # Only print for first sample
            print(f"\n=== DEBUG: Sample 0 ===")
            print(f"Ligand SMILES: {lig_smi}")
            print(f"Mutant ID: {mut_id}")

        # Check/Update Cache for Ligand Features
        if lig_smi in ligand_cache:
            lig_inter, lig_intra = ligand_cache[lig_smi]
        else:
            lig_inter = generate_lig_inter_features(lig_smi)
            lig_intra = generate_lig_intra_features(lig_smi)
            
            if lig_inter is not None and lig_intra is not None:
                ligand_cache[lig_smi] = (lig_inter, lig_intra)

        if idx == 0:  # Only print for first sample
            print(f"lig_inter shape: {lig_inter.shape if lig_inter is not None else None}")
            print(f"lig_intra shape: {lig_intra.shape if lig_intra is not None else None}")
            print(f"=== END DEBUG ===\n")
        
        if any(f is None for f in [lig_inter, lig_intra]):
            continue
        
        lig_inter_list.append(lig_inter)
        lig_intra_list.append(lig_intra)
        mutant_id_list.append(mut_id)
        valid_indices.append(idx)
        
    print(f"  Successfully generated features for {len(valid_indices)} samples")
    
    feature_arrays = {
        'lig_inter': np.array(lig_inter_list),
        'lig_intra': np.array(lig_intra_list),
        'mutant_id': np.array(mutant_id_list),
        'valid_indices': valid_indices
    }
    
    return feature_arrays



# === Neural Network Model ===
def build_model(input_shapes):
    """Construct and compile a multi-input, multi-output feed-forward neural network with embedding layer"""
    # Define inputs
    mutant_input = Input(shape=(1,), name='mutant_id')
    inter_input = Input(shape=(input_shapes['lig_inter'],), name='lig_inter')
    intra_input = Input(shape=(input_shapes['lig_intra'],), name='lig_intra')
    
    # Embedding for mutant ID
    mutant_embedding = Embedding(
        input_dim=input_shapes['num_mutants'], 
        output_dim=16, 
        name='mutant_embedding'
    )(mutant_input)
    mutant_embedding = Flatten()(mutant_embedding)
    
    # Concatenate all inputs
    concat = Concatenate()([mutant_embedding, inter_input, intra_input])
    
    # Dense layers
    x = Dense(512, activation='relu')(concat)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(512, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(128, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    
    # Output heads
    activity_output = Dense(1, name='activity_output')(x)
    docking_output = Dense(1, name='docking_output')(x)
    
    # Build and compile model
    model = Model(
        inputs=[mutant_input, inter_input, intra_input], 
        outputs=[activity_output, docking_output]
    )
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={'activity_output': 'mean_squared_error', 'docking_output': 'mean_squared_error'},
        loss_weights={'activity_output': 1.0, 'docking_output': 0.7},
        metrics={'activity_output': ['mae', 'mse'], 'docking_output': ['mae', 'mse']}
    )
    return model


def get_ligand_features(smiles_series):
    """Helper function to get ligand features from SMILES"""
    lig_inter_list = []
    lig_intra_list = []
    valid_idx = []
    
    for idx, smi in enumerate(smiles_series):
        lig_inter = generate_lig_inter_features(smi)
        lig_intra = generate_lig_intra_features(smi)
        
        if lig_inter is not None and lig_intra is not None:
            lig_inter_list.append(lig_inter)
            lig_intra_list.append(lig_intra)
            valid_idx.append(idx)
    
    return (
        np.array(lig_inter_list) if lig_inter_list else np.array([]),
        np.array(lig_intra_list) if lig_intra_list else np.array([]),
        valid_idx
    )


# ============================================================================
# MAIN SCRIPT
# ============================================================================

def main():
    print("\n" + "="*80)
    print("STAGE 1: GENERATE FEATURES WITH MUTANT INTEGER IDS")
    print("="*80)
    

    # Encode mutations as integers (0 to N-1)
    mutation_names = mutant_valid.unique()
    mutant_encoder = LabelEncoder()
    mutant_encoder.fit(mutation_names)
    
    mutant_mapping = dict(zip(mutant_encoder.classes_, mutant_encoder.transform(mutant_encoder.classes_)))
    
    print("\nMutant ID Mapping:")
    for name, mid in mutant_mapping.items():
        print(f"  {name} -> {mid}")
        
    num_mutants = len(mutant_mapping) + 1  # +1 for safety buffer
    print(f"Total unique mutants: {len(mutant_mapping)}")

    # Encode the training data
    mutant_ids_train = mutant_encoder.transform(mutant_valid)
    
    # Generate features ONCE for the training set
    feature_dict = generate_hierarchical_features(ligand_smiles_valid, mutant_ids_train)
    
    # Common valid indices
    common_valid_indices = sorted(feature_dict['valid_indices'])
    
    # Get y_train
    y_train1 = activity_values_valid.iloc[common_valid_indices].values
    y_train1 = np.log1p(y_train1)  # use log1p (log(1+x)) to avoid errors with 0 values

    y_scaler1 = StandardScaler()
    y_train_scaled1 = y_scaler1.fit_transform(y_train1.reshape(-1, 1)).flatten()

    # Get y_train2 for docking scores
    y_train2 = activity_values2_valid[common_valid_indices]

    y_scaler2 = StandardScaler()
    y_train_scaled2 = y_scaler2.fit_transform(y_train2.reshape(-1, 1)).flatten()

    print(f"\nTraining samples: {len(y_train1)}")
    
    # ===== STAGE 2: TRAIN FEED-FORWARD MODEL =====
    print("\n" + "="*80)
    print("STAGE 2: TRAIN FEED-FORWARD NEURAL NETWORK MODEL")
    print("="*80)
    
    # Normalize Ligand Features
    scalers = {}
    scaled_features = {}
    
    for key in ['lig_inter', 'lig_intra']:
        scalers[key] = StandardScaler()
        scaled_features[key] = scalers[key].fit_transform(feature_dict[key])
    
    # Mutant ID does not need scaling
    scaled_features['mutant_id'] = feature_dict['mutant_id']
    
    # Build model
    feature_dims = {
        'lig_inter': scaled_features['lig_inter'].shape[1],
        'lig_intra': scaled_features['lig_intra'].shape[1],
        'num_mutants': num_mutants + 5  # Buffer
    }
    
    model = build_model(feature_dims)
    model.summary()
        
    checkpoint = ModelCheckpoint(
        'feedforward_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=30,
        restore_best_weights=True,
        verbose=1
    )
    
    print(f"\nTraining feed-forward model...")
    history = model.fit(
        x=[
            scaled_features['mutant_id'],
            scaled_features['lig_inter'],
            scaled_features['lig_intra'],
        ],
        y={
            'activity_output': y_train_scaled1,
            'docking_output': y_train_scaled2
        },
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        callbacks=[early_stop, checkpoint],
        verbose=1
    )
    
    print(f"\n✓ Training complete! Best val_loss: {min(history.history['val_loss']):.4f}")
    
    # Save encoder and scalers
    import pickle
    with open('mutant_encoder.pkl', 'wb') as f:
        pickle.dump(mutant_encoder, f)
    with open('mutant_mapping.pkl', 'wb') as f:
        pickle.dump(mutant_mapping, f)
    with open('feature_scalers.pkl', 'wb') as f:
        pickle.dump(scalers, f)
    with open('y_scalers.pkl', 'wb') as f:
        pickle.dump({'y_scaler1': y_scaler1, 'y_scaler2': y_scaler2}, f)
    
    # Save mutation profiles
    unique_mutation_profiles.to_csv('mutation_profiles.csv', index=False)
    
   
    # Plot training history
    print("\n" + "="*80)
    print("PLOTTING TRAINING HISTORY")
    print("="*80)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(history.history['loss'], label='Train Loss', linewidth=2, color='#2E86AB')
    axes[0].plot(history.history['val_loss'], label='Val Loss', linewidth=2, color='#A23B72')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title('Feed-Forward Model - Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(history.history['activity_output_mae'], label='Train Activity MAE', linewidth=2, color='#2E86AB')
    axes[1].plot(history.history['val_activity_output_mae'], label='Val Activity MAE', linewidth=2, color='#A23B72')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].set_title('Feed-Forward Model - MAE', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Training history plot saved: training_history.png")
    


if __name__ == "__main__":
    print("\n" + "="*80)
    print("STARTING FEED-FORWARD MODEL EXECUTION")
    print("="*80)
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)