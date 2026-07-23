import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, MolSurf, GraphDescriptors, Fragments
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit import DataStructs
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from numpy.linalg import norm
from loguru import logger
import logging
import sys

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate, Multiply, LSTM, GRU, Bidirectional, LeakyReLU, TimeDistributed
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from rdkit import RDLogger

import os
import tensorflow as tf
import pickle

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
    "adv_physchem6d2_{time}.txt",
    rotation="500 MB",
    retention="10 days",
    compression="zip",
    level="DEBUG"
)

print("="*80)
print("RNN-LSTM-KAN INTEGRATED HIERARCHICAL MODEL FOR ADVANCED PHYSICOCHEMICAL DESCRIPTOR GENERATION (6d2)")
print("Sequential Training: FULL -> ATP_POCKET -> P_LOOP -> C_HELIX -> DEL19 -> HINGE_LOOP -> DFG_A_LOOP -> HRD_CAT")
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

#Substructure smiles for feature capture of mutation protein
#Mechanistic reaction order for EGFR: full -> ATP pocket -> P-loop -> C-helix ->
#19-deletions -> hinge loop (T790M/C797S) -> A-loop DFG -> HRD catalytic motif
ligand_smiles = df_train['smiles']
full_smiles = df_train['smiles_full_sequence_egfr_manual']
mutation_smiles = df_train['smiles_sequence_atp_ pocket']
mut_p_loop = df_train['smiles_sequence_p_loop_constant']
mut_helix = df_train['smiles_sequence_c_helix_constant']
mut_19_del = df_train['smiles_sequence_19_deletions']
mut_hinge_loop = df_train['smiles_sequence_hinge_loop_t790m_c797s']
mut_dfg_a_loop = df_train['smiles_sequence_a_loop_dfg']
mut_hrd_cat = df_train['smiles_sequence_hrd_constant']

mutant = df_train['tkd']

activity_values = df_train['standard value'] #y_train1
docking_values = df_train['dock'] #y_train2



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
    print("\n✗ ERROR: No complete samples found!")
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



#Assumptions:
# 1. KAN Gausian RBF basis functions can effectively capture non-linear relationships in Hierachical layers
# 1. Navier-stokes fluid mechanics gradients has significant feature capture at RNN LTSM layers
# 2. KAN layers with non-linear gradient basis can better approximate complex non-linear relationships in molecular data
# 3. KAN layers are independent and can be integrated into existing hierachical and RNN LTSM layers 
# 4. Navier Stokes equations allow sinusoidal periodic sequence basis approximate for EGFR mechanistic catalytic activity 


# ============================================================================
# KAN LAYER IMPLEMENTATION (Efficient-KAN style with Gaussian RBF)
# ============================================================================

# φ(x) = wᵦ · SiLU(x) + wₛ · spline(x)
#        ↑              ↑
#    base function   learnable B-spline

# y = SiLU(x)·W_base + Σₖ₌₁ᴳ cₖ·ψₖ(x) , where Gaussian RBF ψₖ(x) = exp(-(x - μₖ)²/σ²)

class KANLayer(tf.keras.layers.Layer):

    def __init__(self, out_features, grid_size=20, grid_range=[-2.0, 2.0], **kwargs):
        super(KANLayer, self).__init__(**kwargs)
        self.out_features = out_features
        self.grid_size = grid_size
        self.grid_min, self.grid_max = grid_range
        
    def build(self, input_shape):
        in_features = input_shape[-1]
        
        # Grid initialization (Fixed grid points)
        self.grid = tf.linspace(self.grid_min, self.grid_max, self.grid_size)
        self.grid = tf.cast(self.grid, dtype=tf.float32) # (grid_size,)
        
        # Use a non-trainable weight for grid centers so it persists
        self.mu = self.add_weight(
            name='mu',
            shape=(self.grid_size,),
            initializer=tf.keras.initializers.Constant(self.grid.numpy()), # Use numpy value
            trainable=False 
        )
        
        # Sigma (bandwidth)
        spacing = (self.grid_max - self.grid_min) / (self.grid_size - 1)
        self.sigma = spacing
        
        # Base Weights (Linear approximation part)
        self.base_weight = self.add_weight(
            name='base_weight',
            shape=(in_features, self.out_features),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Spline Weights (Coefficients for RBFs)
        self.spline_weight = self.add_weight(
            name='spline_weight',
            shape=(in_features, self.grid_size, self.out_features),
            initializer='glorot_uniform',
            trainable=True
        )
        
        super(KANLayer, self).build(input_shape)

    def call(self, x):
        # x shape: (..., in_features)
        
        # 1. Base Feature Transformation (SiLU activation)
        base = tf.nn.silu(x)
        # Use tensordot or einsum to handle broadcasting safely
        # '...i, io -> ...o'
        base_out = tf.einsum('...i,io->...o', base, self.base_weight)
        
        # 2. Spline Part (RBF Expansion)
        # Expand x: (..., in) -> (..., in, 1)
        x_expanded = tf.expand_dims(x, -1)
        
        # Compute distance to grid centers (mu)
        # (..., in, 1) - (grid,) -> (..., in, grid)
        diff = x_expanded - self.mu
        
        # RBF Basis functions (Gaussian)
        # exp(-(x - mu)^2 / sigma^2)
        basis = tf.exp(-tf.math.pow(diff / self.sigma, 2)) 
        
        # Compute spline output
        # '...ig, igo -> ...o' 
        spline_out = tf.einsum('...ig,igo->...o', basis, self.spline_weight)
        
        # Final output
        return base_out + spline_out
        
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.out_features,)

    def get_config(self):
        config = super(KANLayer, self).get_config()
        config.update({
            'out_features': self.out_features,
            'grid_size': self.grid_size,
            'grid_range': [self.grid_min, self.grid_max]
        })
        return config

# ============================================================================
# FOURIER KAN LAYER (NAVIER STOKES SINUSOID BASIS)
# ============================================================================

# φ(x) = wᵦ · SiLU(x) + wₛ · spline(x)
#        ↑              ↑
#    base function   learnable B-spline

# y = Σₖ₌₁ᴳ [aₖ·cos(kx) + bₖ·sin(kx)] + bias


class FourierKANLayer(tf.keras.layers.Layer):

    def __init__(
        self,
        out_features,
        grid_size=5,
        add_bias=True,
        domain="[-pi, pi]",   
        **kwargs
    ):
        super(FourierKANLayer, self).__init__(**kwargs)
        self.out_features = out_features
        self.grid_size = grid_size
        self.add_bias = add_bias
        self.domain = domain
        
    def build(self, input_shape):
        in_features = input_shape[-1]
        
        limit = 1.0 / (np.sqrt(in_features) * np.sqrt(self.grid_size))
        
        self.fouriercoeffs = self.add_weight(
            name="fouriercoeffs",
            shape=(2, self.out_features, in_features, self.grid_size),
            initializer=tf.keras.initializers.RandomNormal(stddev=limit),
            trainable=True
        )
        
        if self.add_bias:
            self.bias = self.add_weight(
                name="bias",
                shape=(self.out_features,),
                initializer="zeros",
                trainable=True
            )
            
        super().build(input_shape)

    def call(self, x):

        if self.domain == "[-pi, pi]":
            # assume x ∈ [0,1] or arbitrary → scale
            x = tf.clip_by_value(x, 0.0, 1.0)
            x = (x * 2.0 - 1.0) * np.pi

        elif self.domain == "[0, 2pi]":
            x = tf.clip_by_value(x, 0.0, 1.0)
            x = x * (2.0 * np.pi)

        original_shape = tf.shape(x)
        total_batch_size = tf.reduce_prod(original_shape[:-1])
        in_feats = original_shape[-1]
        
        x_flat = tf.reshape(x, (total_batch_size, in_feats))
        x_rshp = tf.reshape(x_flat, (total_batch_size, 1, in_feats, 1))
        
        k = tf.reshape(
            tf.range(1, self.grid_size + 1, dtype=tf.float32),
            (1, 1, 1, self.grid_size)
        )
        
        angles = x_rshp * k
        
        c = tf.cos(angles)
        s = tf.sin(angles)
        
        c = tf.reshape(c, (1, total_batch_size, in_feats, self.grid_size))
        s = tf.reshape(s, (1, total_batch_size, in_feats, self.grid_size))
        
        cs = tf.concat([c, s], axis=0)
        
        y = tf.einsum("dbik,djik->bj", cs, self.fouriercoeffs)
        
        if self.add_bias:
            y = y + self.bias
            
        final_shape = tf.concat(
            [original_shape[:-1], [self.out_features]], axis=0
        )
        y = tf.reshape(y, final_shape)
        
        return y

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.out_features,)
        
    def get_config(self):
        config = super().get_config()
        config.update({
            "out_features": self.out_features,
            "grid_size": self.grid_size,
            "add_bias": self.add_bias,
            "domain": self.domain
        })
        return config

    
# ============================================================================
# FEATURE GENERATION FUNCTIONS 
# ============================================================================

#Assumptions:
#1. Intermolecular forces dominate interactions between ligand and mutation while intramolecular forces dominate within own ligand and own mutation
#2. Custom Interaction features using intermolecular and intramolecular forces corelates a better representation of activity between ligand and mutations.
#3. Similarity and fingerprint metrics further finetunes the repressentation.  

#Priority of descriptors to capture intermolecular features between ligand and mutation

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
        features.append(MolSurf.TPSA(mol))#The sum of the surface areas of all polar atoms (mostly oxygen and nitrogen) and their attached hydrogens.High TPSA → more polar, less membrane permeable, more soluble, Low TPSA → less polar, more membrane permeable (good for oral drugs)
        
        features.append(MolSurf.LabuteASA(mol))#An approximation of the total solvent-accessible surface area (SASA) of the molecule, calculated using Labute’s algorithm. #Reflects molecular size and hydrophobic surface exposure
         #13 #Ai=Si⋅Pi, S = 4πr^2i , Pi=1−j∏(1−fij
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
        
        single_bonds = sum(1 for bond in mol.GetBonds() if bond.GetBondTypeAsDouble() == 1.0)
        double_bonds = sum(1 for bond in mol.GetBonds() if bond.GetBondTypeAsDouble() == 2.0)
        triple_bonds = sum(1 for bond in mol.GetBonds() if bond.GetBondTypeAsDouble() == 3.0)
        aromatic_bonds = sum(1 for bond in mol.GetBonds() if bond.GetIsAromatic())
        features.extend([single_bonds, double_bonds, triple_bonds, aromatic_bonds])
        
        avg_bond_order = np.mean([bond.GetBondTypeAsDouble() for bond in mol.GetBonds()]) if num_bonds > 0 else 0
        features.append(avg_bond_order)
        
        #Rigidity (May Overlap)
        features.append(Lipinski.NumRotatableBonds(mol))
        features.append(Lipinski.RingCount(mol))
        features.append(Lipinski.NumAromaticRings(mol))
        
        rigid_bonds = sum(1 for bond in mol.GetBonds() if bond.IsInRing())
        fraction_rigid = rigid_bonds / num_bonds if num_bonds > 0 else 0
        features.append(fraction_rigid)
        
        # Pi-Pi bonding 
        features.append(Descriptors.NumAromaticCarbocycles(mol))
        features.append(Descriptors.NumAromaticHeterocycles(mol))
        
        #Hybridization (May Overlap)
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
    
    return {
        'cosine_similarity': cosine_sim,
        'sine_dissimilarity': sine_of_angle,
        'dot_product': np.dot(vec1, vec2)
    }


def calculate_fp_metrics(smiles1, smiles2): #input smiles, returns dict with rdkit datastructs similarity
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    
    if mol1 is None or mol2 is None:
        return {'dice_sim': 0.0, 'tanimato': 0.0}
    
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
    
    dice_sim = DataStructs.DiceSimilarity(fp1, fp2)
    tanimato = DataStructs.TanimotoSimilarity(fp1, fp2)
    
    return {
        'dice_sim': dice_sim,
        'tanimato': tanimato,
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


def generate_final_interaction_features(lig_smiles, mut_smiles): # fingerprints
    features = []
    
    fp_inter_metrics = calculate_fp_metrics(lig_smiles, mut_smiles)
    features.extend([fp_inter_metrics['dice_sim'], fp_inter_metrics['tanimato']])
    
    return np.array(features)


def generate_custom_features(lig_inter, mut_inter, lig_intra, mut_intra): 
    """Generate custom intermolecular and intramolecular features with safe division"""
    lig_mut_inter = []
    lig_mut_intra = []
    lig_mut_mix_inter_intra = []
    
    # H attraction ligand , H = lig_hbd . mut_hba / mut_hbd 
    # (#assumption: ligand moves to mut (mut is fixed position), lig hbd and mut hba attracts ligand), 
    # mut hbd repels favouring intra bond within mut, ignoring intra bond repelling from ligand
    H_linear_lipinski = safe_divide(lig_inter[0] * mut_inter[1], mut_inter[0], default=0.0)
    lig_mut_inter.append(H_linear_lipinski)
    
    H_linear_total = safe_divide(lig_inter[4] * mut_inter[5], mut_inter[4], default=0.0)
    lig_mut_inter.append(H_linear_total)
    
    # H attraction ligand , H = lig_hbd . mut_hba / mut_hbd with weighted mut bond path (Kappa)
    # (#assumption: ligand moves to mut (mut is fixed position), lig hbd and mut hba attracts ligand), 
    # mut hbd repels favouring intra bond within mut
    H_path = safe_divide(safe_divide(lig_inter[0] * mut_inter[1], mut_inter[0], default=0.0), mut_intra[21], default=0.0)
    lig_mut_mix_inter_intra.append(H_path)
    
    # Streght H bond in intermolecular lig to mut minus mut intra bond within mut Lig(x1,y1) Mut(x2,y2)
    # total attraction H_stregth , Lig(x1y1) Mut(x2,y2) , (lig_x1 * mut_y2 / lig_x2) + (lig_x2 * mut_y1 / mut_y2)
    # inter bond attarct x1y2 , intra bond forming assumed as repelled, x1/y1 , assumed no repelling inter H bonds
    H_strength = safe_divide(lig_inter[0] * mut_inter[1], lig_inter[1], default=0.0) + safe_divide(lig_inter[1] * mut_inter[0], mut_inter[1], default=0.0)
    lig_mut_inter.append(H_strength)
    
    H_strength_total = safe_divide(lig_inter[4] * mut_inter[5], lig_inter[4], default=0.0) + safe_divide(lig_inter[5] * mut_inter[4], mut_inter[5], default=0.0)
    lig_mut_inter.append(H_strength_total)
    
    # Lig donating stregght + Mut accepting Stregth , ligand movving to mut
    H_frac_lipinski = safe_divide(lig_inter[0], lig_inter[1], default=0.0) + safe_divide(mut_inter[1], mut_inter[0], default=0.0)
    lig_mut_inter.append(H_frac_lipinski)
    
    H_frac_total = safe_divide(lig_inter[4], lig_inter[5], default=0.0) + safe_divide(mut_inter[5], mut_inter[4], default=0.0)
    lig_mut_inter.append(H_frac_total)
    

    #using max positive and max negative charge, and length and size is simple number of bonds  (q1q2/r2)
    # Attraction opp site charge lig(q1/r1) * mut(q2/r2), q1 is max positive and q2 is max neg
    # size options include: Molwt, number of bonds, Euclidean distance . radius of gyration (rdMolDescriptors.CalcRadiusOfGyration(mol))

    # Assumption: non moving mutant, only ligand moving to mutant through attraction charge Only, (taking max abs postive and min ngeative)
    # only Attraction intermolecular forces, assuming no intrabond attraction within molecule. Assumed no repelling intermolecule same charge
    #A c_linear q1 pos to q2 neg / r1r2 
    # B c_linear q1 neg to q2 pos/r1r2
    #total & ratio

    # assuming got positive charges ligand and negative charge mut with weighted size sp3
    c_linear1_size1 = safe_divide(lig_inter[6], lig_intra[14], default=0.0) * safe_divide(mut_inter[7], mut_intra[14], default=0.0)
    lig_mut_mix_inter_intra.append(c_linear1_size1)
    
    c_linear2_size1 = safe_divide(lig_inter[7], lig_intra[14], default=0.0) * safe_divide(mut_inter[6], mut_intra[14], default=0.0)
    lig_mut_mix_inter_intra.append(c_linear2_size1)
    
    c_total = (c_linear1_size1 ** 2) + (c_linear2_size1 ** 2) #bringing out magnitude of each attarction parts
    lig_mut_mix_inter_intra.append(c_total)
    
    #difference between pos lig neg mut to neg mut pos lig
    c_diff = ((lig_inter[6]) - (mut_inter[7])) - ((mut_inter[6]) - (lig_inter[7]))
    lig_mut_inter.append(c_diff)
    
    #difference between pos lig neg mut to neg mut pos lig
    c_tpsa_diff = lig_inter[11] - mut_inter[11]
    lig_mut_inter.append(c_tpsa_diff)
    
    c_crip_logh = lig_inter[17] - mut_inter[17]
    lig_mut_inter.append(c_crip_logh)
    
    frac_tpsa_logH = safe_divide(lig_inter[11] * mut_inter[11], lig_inter[17] * mut_inter[17], default=0.0)
    lig_mut_inter.append(frac_tpsa_logH)
    
    #pi-pi stacking ratio
    pi_pi_ratio1 = safe_divide(lig_inter[21] + lig_inter[22] + mut_inter[21] + mut_inter[22], lig_intra[15] + mut_intra[15], default=0.0)
    lig_mut_mix_inter_intra.append(pi_pi_ratio1)
    
    pi_pi_ratio2 = safe_divide(lig_inter[21] + lig_inter[22] + mut_inter[21] + mut_inter[22], lig_intra[22] + mut_intra[22], default=0.0)
    lig_mut_mix_inter_intra.append(pi_pi_ratio2)
    
    #Bringing out difference between a more rigid/loose ligand 

    #double/triple bond ratio increasing
    # bond rigid total double, triple n aromatic over total num of bonds (tighter intra lig and intra mut strength as a total)
    # bond single (looser intra lig and intra mut strength)
    bond_rigid = safe_divide(lig_intra[2] + lig_intra[3] + lig_intra[4], lig_intra[0], default=0.0) + safe_divide(mut_intra[2] + mut_intra[3] + mut_intra[4], mut_intra[0], default=0.0)
    bond_single = safe_divide(lig_intra[1], lig_intra[0], default=0.0) + safe_divide(mut_intra[1], mut_intra[0], default=0.0)
    bond_diff = (bond_single - bond_rigid) ** 2
    lig_mut_intra.append(bond_diff)
    
    #spsp2/sp3 ratio
    # fraction of spsp2/sp3 between ligand and mutant
    # bigger difference indicate mutants more loose, ligands are same
    hybridisation_lig = safe_divide(lig_intra[12] + lig_intra[13], lig_intra[14] + lig_intra[12] + lig_intra[13], default=0.0)
    hybridisation_mut = safe_divide(mut_intra[12] + mut_intra[13], mut_intra[14] + mut_intra[12] + mut_intra[13], default=0.0)
    hybridisation_diff = (hybridisation_mut - hybridisation_lig) ** 2
    lig_mut_intra.append(hybridisation_diff)
    
    bertz_ratio = safe_divide(lig_intra[21], mut_intra[21], default=0.0)
    lig_mut_intra.append(bertz_ratio)
    
    return lig_mut_inter, lig_mut_intra, lig_mut_mix_inter_intra

def generate_hierarchical_features(ligand_smiles_series, mutation_smiles_series):
    lig_inter_list = []
    mut_inter_list = []
    inter_interaction_list = []
    lig_intra_list = []
    mut_intra_list = []
    intra_interaction_list = []
    lig_mut_mix_inter_intra_list = []
    fp_interaction_list = []
    
    valid_indices = []
    
    print("\nGenerating hierarchical features...")

    ligand_cache = {}
    mutation_cache = {}
    interaction_cache = {}
    
    for idx, (lig_smi, mut_smi) in enumerate(zip(ligand_smiles_series, mutation_smiles_series)):
        if idx % 50 == 0:
            print(f"  Processing sample {idx}/{len(ligand_smiles_series)}...")
        
        if lig_smi in ligand_cache:
            lig_inter, lig_intra = ligand_cache[lig_smi]
        else:
            lig_inter = generate_lig_inter_features(lig_smi)
            lig_intra = generate_lig_intra_features(lig_smi)
            ligand_cache[lig_smi] = (lig_inter, lig_intra)

        if mut_smi in mutation_cache:
            mut_inter, mut_intra = mutation_cache[mut_smi]
        else:
            mut_inter = generate_mut_inter_features(mut_smi)
            mut_intra = generate_mut_intra_features(mut_smi)
            mutation_cache[mut_smi] = (mut_inter, mut_intra)

        if any(f is None for f in [lig_inter, mut_inter, lig_intra, mut_intra]):
            continue
        
        pair_key = (lig_smi, mut_smi)
        
        if pair_key in interaction_cache:
            lig_mut_mix_inter_intra, inter_interaction, intra_interaction, final_fp_interaction = interaction_cache[pair_key]
        else:
            lig_mut_inter, lig_mut_intra, lig_mut_mix_inter_intra = generate_custom_features(
                lig_inter, mut_inter, lig_intra, mut_intra
            )

            inter_interaction = generate_inter_interaction_features(lig_inter, mut_inter)
            intra_interaction = generate_intra_interaction_features(lig_intra, mut_intra)
            
            if len(lig_mut_inter) > 0:
                inter_interaction = np.concatenate([np.array(lig_mut_inter), inter_interaction])
            
            if len(lig_mut_intra) > 0:
                intra_interaction = np.concatenate([np.array(lig_mut_intra), intra_interaction])

            final_fp_interaction = generate_final_interaction_features(lig_smi, mut_smi)
            
            interaction_cache[pair_key] = (lig_mut_mix_inter_intra, inter_interaction, intra_interaction, final_fp_interaction)
    
        # Single comprehensive logging before appending
        logger.debug(f"Sample {idx} - Generated features: lig_inter shape={lig_inter.shape}, "
                   f"mut_inter shape={mut_inter.shape}, "
                   f"lig_intra shape={lig_intra.shape}, "
                   f"mut_intra shape={mut_intra.shape}, "
                   f"inter_interaction shape={inter_interaction.shape}, "
                   f"intra_interaction shape={intra_interaction.shape}, "
                   f"lig_mut_mix_inter_intra shape={len(lig_mut_mix_inter_intra)}, "
                   f"final_fp_interaction shape={final_fp_interaction.shape}")
        
        lig_inter_list.append(lig_inter)
        mut_inter_list.append(mut_inter)
        inter_interaction_list.append(inter_interaction)
        lig_intra_list.append(lig_intra)
        mut_intra_list.append(mut_intra)
        intra_interaction_list.append(intra_interaction)
        fp_interaction_list.append(final_fp_interaction)
        lig_mut_mix_inter_intra_list.append(np.array(lig_mut_mix_inter_intra))
        
        valid_indices.append(idx)
        
    print(f"  Successfully generated features for {len(valid_indices)} samples")
    
    return {
        'lig_inter': np.array(lig_inter_list),
        'mut_inter': np.array(mut_inter_list),
        'inter_interaction': np.array(inter_interaction_list),
        'lig_intra': np.array(lig_intra_list),
        'mut_intra': np.array(mut_intra_list),
        'intra_interaction': np.array(intra_interaction_list),
        'lig_mut_mix_inter_intra': np.array(lig_mut_mix_inter_intra_list),
        'final_fp_interaction': np.array(fp_interaction_list),
        'valid_indices': valid_indices
    }

#Forward Neural Network based on Priority Hierachy connecting and gating weights
#Higher-priority features control what lower-priority features contribute, creating a learned feature hierarchy rather than treating all inputs equally

#Rationale on Model (with KAN Integration):
#1. Concatenation to keep highest priority features as dominant baseline 
#2. Multiplification to filter and gate features on top
#3. LeakyReLU for initial transformations
#4. KANLayers for capturing non-linear feature interactions and embeddings
#5. Tanh for KAN output scaling (optional, if within KAN)
#6. Sigmoid for gating mechanisms

def build_priority_hierarchical_model(feature_dims):
    """
    Build hierarchical model with priority-based feature processing and KAN layers.
    Logs the shape transformations and mathematical operations at each layer.
    """
    logger.info("="*80)
    logger.info("BUILDING PRIORITY HIERARCHICAL MODEL (KAN INTEGRATED)")
    logger.info("="*80)
    
    # Log input dimensions
    logger.info("\n--- INPUT LAYER DIMENSIONS ---")
    for key, dim in feature_dims.items():
        logger.info(f"  {key}: ({dim},) -> Vector of length {dim}")
    
    # Define inputs
    final_interaction_input = Input(shape=(feature_dims['final_fp_interaction'],), name='final_fp_interaction') #Fingerprint features
    lig_mut_mix_inter_intra_input = Input(shape=(feature_dims['lig_mut_mix_inter_intra'],), name='lig_mut_mix_inter_intra') #Custom features
    inter_interaction_input = Input(shape=(feature_dims['inter_interaction'],), name='inter_interaction') #Similarity features
    intra_interaction_input = Input(shape=(feature_dims['intra_interaction'],), name='intra_interaction') 
    mut_inter_input = Input(shape=(feature_dims['mut_inter'],), name='mut_inter') #Descriptor features
    lig_inter_input = Input(shape=(feature_dims['lig_inter'],), name='lig_inter')
    mut_intra_input = Input(shape=(feature_dims['mut_intra'],), name='mut_intra')
    max_lig_intra_dim = feature_dims.get('lig_intra', 0)
    lig_intra_input = Input(shape=(max_lig_intra_dim,), name='lig_intra')
    
    logger.info("\n" + "="*80)
    logger.info("PRIORITY 1: Final FP Interaction + Lig-Mut Mix Inter-Intra (using KAN)")
    logger.info("="*80)
    
    # Priority 1 - Final branch
    logger.info(f"\n[Final Branch]")
    logger.info(f"  Input: (batch, {feature_dims['final_fp_interaction']})")
    
    final_branch = Dense(32, kernel_initializer='he_normal', name='final_dense1')(final_interaction_input)
    final_branch = LeakyReLU(alpha=0.1, name='final_leaky1')(final_branch)
    logger.info(f"  Dense(32) + LeakyReLU: -> (batch, 32)")
    
    # KAN for embedding
    final_emb = KANLayer(out_features=8, name='final_emb_kan')(final_branch)
    logger.info(f"  KANLayer(8): Captures non-linear spline-based transformations -> (batch, 8)")
    
    # Priority 1 - Mix inter-intra branch 
    logger.info(f"\n[Lig-Mut Mix Inter-Intra Branch]")
    logger.info(f"  Input: (batch, {feature_dims['lig_mut_mix_inter_intra']})")
    
    lig_mut_mix_inter_intra_branch = Dense(8, kernel_initializer='he_normal', name='mix_inter_intra_dense')(lig_mut_mix_inter_intra_input)
    lig_mut_mix_inter_intra_branch = LeakyReLU(alpha=0.1, name='mix_inter_intra_leaky')(lig_mut_mix_inter_intra_branch)
    
    lig_mut_mix_inter_intra_branch_emb = KANLayer(out_features=4, name='mix_emb_kan')(lig_mut_mix_inter_intra_branch)
    logger.info(f"  KANLayer(4): -> (batch, 4)")
    
    # Combine Priority 1
    priority1_combined = Concatenate(name='priority1_combined')([final_emb, lig_mut_mix_inter_intra_branch_emb])
    logger.info(f"\n[Priority 1 Combined] -> (batch, 12)")
    
    # Priority 2
    logger.info("\n" + "="*80)
    logger.info("PRIORITY 2: Inter-molecular Interactions (with gating and KAN)")
    logger.info("="*80)
    
    inter_interact_branch = Dense(48, kernel_initializer='he_normal', name='inter_dense1')(inter_interaction_input)
    inter_interact_branch = LeakyReLU(alpha=0.1, name='inter_leaky1')(inter_interact_branch)
    inter_interact_branch = BatchNormalization(name='inter_bn1')(inter_interact_branch)
    
    # Gating mechanism
    inter_gate = Dense(48, activation='sigmoid', kernel_initializer='glorot_uniform', name='inter_gate')(
        Dense(24, activation='relu', name='inter_gate_pre')(priority1_combined)
    )
    logger.info(f"\n[Gating Mechanism] Using Priority 1 to gate Priority 2")
    
    inter_gated = Multiply(name='inter_gating')([inter_interact_branch, inter_gate])
    inter_emb = KANLayer(out_features=12, name='inter_emb_kan')(inter_gated)
    logger.info(f"  KANLayer(12): -> (batch, 12)")
    
    priority1_2_combined = Concatenate(name='priority1_2_combined')([priority1_combined, inter_emb])
    logger.info(f"\n[Priority 1+2 Combined] -> (batch, 24)")
    
    # Priority 3
    logger.info("\n" + "="*80)
    logger.info("PRIORITY 3: Intra-molecular Interactions (with gating and KAN)")
    logger.info("="*80)
    
    intra_interact_branch = Dense(48, kernel_initializer='he_normal', name='intra_dense1')(intra_interaction_input)
    intra_interact_branch = LeakyReLU(alpha=0.1, name='intra_leaky1')(intra_interact_branch)
    intra_interact_branch = BatchNormalization(name='intra_bn1')(intra_interact_branch)
    
    intra_gate = Dense(48, activation='sigmoid', kernel_initializer='glorot_uniform', name='intra_gate')(
        Dense(24, activation='relu', name='intra_gate_pre')(priority1_2_combined)
    )
    
    intra_gated = Multiply(name='intra_gating')([intra_interact_branch, intra_gate])
    intra_emb = KANLayer(out_features=12, name='intra_emb_kan')(intra_gated)
    logger.info(f"  KANLayer(12): -> (batch, 12)")
    
    priority1_2_3_combined = Concatenate(name='priority1_2_3_combined')([priority1_2_combined, intra_emb])
    logger.info(f"\n[Priority 1+2+3 Combined] -> (batch, 36)")
    
    # Priority 4-5 (Inter features)
    logger.info("\n" + "="*80)
    logger.info("PRIORITY 4-5: Individual Inter-molecular Features (Mutation & Ligand)")
    logger.info("="*80)
    
    # Mutation inter
    mut_inter_branch = BatchNormalization()(LeakyReLU(alpha=0.1)(Dense(32)(mut_inter_input)))
    mut_inter_gate = Dense(32, activation='sigmoid')(Dense(16, activation='relu')(priority1_2_3_combined))
    mut_inter_gated = Multiply()([mut_inter_branch, mut_inter_gate])
    mut_inter_emb = KANLayer(out_features=8, name='mut_inter_emb_kan')(mut_inter_gated)
    
    # Ligand inter
    lig_inter_branch = BatchNormalization()(LeakyReLU(alpha=0.1)(Dense(32)(lig_inter_input)))
    lig_inter_gate = Dense(32, activation='sigmoid')(Dense(16, activation='relu')(priority1_2_3_combined))
    lig_inter_gated = Multiply()([lig_inter_branch, lig_inter_gate])
    lig_inter_emb = KANLayer(out_features=8, name='lig_inter_emb_kan')(lig_inter_gated)
    
    logger.info(f"  Mutation & Ligand Inter branches gated and passed through KAN(8)")
    
    # Priority 6-7 (Intra features)
    logger.info("\n" + "="*80)
    logger.info("PRIORITY 6-7: Individual Intra-molecular Features (Mutation & Ligand)")
    logger.info("="*80)
    
    all_combined_so_far = Concatenate()([priority1_2_3_combined, mut_inter_emb, lig_inter_emb])
    
    # Mutation intra
    mut_intra_branch = BatchNormalization()(LeakyReLU(alpha=0.1)(Dense(32)(mut_intra_input)))
    mut_intra_gate = Dense(32, activation='sigmoid')(Dense(16, activation='relu')(all_combined_so_far))
    mut_intra_gated = Multiply()([mut_intra_branch, mut_intra_gate])
    mut_intra_emb = KANLayer(out_features=8, name='mut_intra_emb_kan')(mut_intra_gated)
    
    # Ligand intra
    lig_intra_branch = BatchNormalization()(LeakyReLU(alpha=0.1)(Dense(32)(lig_intra_input)))
    lig_intra_gate = Dense(32, activation='sigmoid')(Dense(16, activation='relu')(all_combined_so_far))
    lig_intra_gated = Multiply()([lig_intra_branch, lig_intra_gate])
    lig_intra_emb = KANLayer(out_features=8, name='lig_intra_emb_kan')(lig_intra_gated)
    
    # Final Integration
    logger.info("\n" + "="*80)
    logger.info("FINAL CONCATENATION AND HIERARCHICAL INTEGRATION")
    logger.info("="*80)
    
    final_features = Concatenate(name='final_features_concat')([
        all_combined_so_far, mut_intra_emb, lig_intra_emb
    ])
    
    # Final KAN integration layers
    x = KANLayer(out_features=128, name='final_integration_kan1')(final_features)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = KANLayer(out_features=64, name='final_integration_kan2')(x)
    
    # Embedding Layer (Critical Output for RNN)
    embedding_layer = Dense(16, kernel_initializer='he_normal', name='embedding_layer')(x)
    embedding_output = LeakyReLU(alpha=0.1, name='embedding_output')(embedding_layer)
    logger.info(f"  Hierarchical Embedding Produced: (batch, 16)")
    
    # Output Heads
    activity_output = Dense(1, activation='linear', name='activity_output')(
        Dense(8, activation='relu')(embedding_output)
    )
    docking_output = Dense(1, activation='linear', name='docking_output')(
        Dense(8, activation='relu')(embedding_output)
    )
    
    model = Model(
        inputs=[final_interaction_input, lig_mut_mix_inter_intra_input, inter_interaction_input, 
                intra_interaction_input, mut_inter_input, lig_inter_input, mut_intra_input, lig_intra_input],
        outputs=[activity_output, docking_output],
        name='priority_hierarchical_model_kan'
    )
    
    model.compile(
        optimizer=Adam(learning_rate=0.003), 
        loss=['mse', 'mse'], 
        loss_weights=[1.0, 0.6], 
        metrics={'activity_output': ['mae', 'mse'], 'docking_output': ['mae', 'mse']}
    )
    
    model.summary(print_fn=logger.info)
    return model

# ============================================================================
# RNN SEQUENTIAL MODEL WITH KAN (MERGED)
# ============================================================================


def build_rnn_sequential_model(embedding_dim, n_timesteps=8):
    """
    Build RNN model with BiLSTM and parallel BiGRU + KAN path.
    Logs the architecture and parameters.
    """
    logger.info("="*80)
    logger.info("BUILDING RNN-KAN SEQUENTIAL MODEL")
    logger.info("="*80)
    
    sequence_input = Input(shape=(n_timesteps, embedding_dim), name='mutation_sequence')
    
    # Path 1: BiLSTM (Original)
    logger.info("\n[Path 1: BiLSTM]")
    lstm_out = Bidirectional(LSTM(128, return_sequences=True, dropout=0.2))(sequence_input)
    lstm_out = BatchNormalization()(lstm_out)
    lstm_out = Bidirectional(LSTM(64, return_sequences=False, dropout=0.2))(lstm_out)
    lstm_out = BatchNormalization()(lstm_out)
    
    # Path 2: BiGRU + Fourier KAN (Updated)
    logger.info("\n[Path 2: BiGRU + Fourier KAN]")
    gru_out1 = Bidirectional(GRU(128, return_sequences=True, dropout=0.2))(sequence_input)
    gru_out1 = BatchNormalization()(gru_out1)
    
    # Fourier KAN Layer 1
    # Using grid_size=5 as per ka_gnn2 default
    kan_out1 = FourierKANLayer(out_features=128, grid_size=5, domain="[-pi, pi]", name='fourier_kan_1')(gru_out1)
    
    # BiGRU 2
    gru_out2 = Bidirectional(GRU(64, return_sequences=False, dropout=0.2))(kan_out1)
    
    # Fourier KAN Layer 2
    kan_out2 = FourierKANLayer(out_features=128, grid_size=5, domain="[-pi, pi]", name='fourier_kan_2')(gru_out2)
    kan_out2 = BatchNormalization()(kan_out2)
    
    # Combine Paths
    logger.info("\n[Combining BiLSTM and BiGRU+FourierKAN paths]")
    combined = Concatenate()([lstm_out, kan_out2])
    
    # Output Heads with Fourier KAN Integration
    x = FourierKANLayer(out_features=128, grid_size=5, domain="[-pi, pi]", name='final_merged_fourier_kan')(combined)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    
    activity_output = Dense(1, activation='linear', name='final_activity_output')(x)
    docking_output = Dense(1, activation='linear', name='final_docking_output')(x)
    
    model = Model(inputs=sequence_input, outputs=[activity_output, docking_output], name='rnn_kan_model')
    
    model.compile(
        optimizer=Adam(learning_rate=0.001), 
        loss=['mse', 'mse'], 
        loss_weights=[1.0, 0.7], 
        metrics={'final_activity_output': ['mae'], 'final_docking_output': ['mae']}
    )
    
    logger.info("RNN-KAN Model Built")
    model.summary(print_fn=logger.info)
    
    return model

# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    print("\n" + "="*80)
    print("STAGE 1: GENERATE FEATURES FOR ALL MUTATION SITES")
    print("="*80)
    
    # Mechanistic reaction order for EGFR: full -> ATP pocket -> P-loop -> C-helix ->
    # 19-deletions -> hinge loop (T790M/C797S) -> A-loop DFG -> HRD catalytic motif
    # This exact order is also used by the predict script and must never diverge from it,
    # since it defines the RNN timestep order.
    mutation_sites = [
        ('FULL_SMILES', full_smiles_valid),
        ('ATP_POCKET', mutation_smiles_valid),
        ('P_LOOP', mut_p_loop_valid),
        ('C_HELIX', mut_helix_valid),
        ('DEL19', mut_19_del_valid),
        ('HINGE_LOOP', mut_hinge_loop_valid),
        ('DFG_A_LOOP', mut_dfg_a_loop_valid),
        ('HRD_CAT', mut_hrd_cat_valid)
    ]
    
    all_feature_dicts = []
    all_valid_indices = []
    
    for site_name, mut_smiles_series in mutation_sites:
        print(f"\n{'='*80}")
        print(f"Processing {site_name}")
        print(f"{'='*80}")
        
        feature_dict = generate_hierarchical_features(ligand_smiles_valid, mut_smiles_series)
        all_feature_dicts.append(feature_dict)
        all_valid_indices.append(set(feature_dict['valid_indices']))
        logger.info(f" Valid indices for {site_name}: {feature_dict['valid_indices']}")
    
    # Find common valid indices
    common_valid_indices = set.intersection(*all_valid_indices)
    common_valid_indices = sorted(list(common_valid_indices))
    logger.info(f"Common valid indices across all sites: {common_valid_indices}")
    
    print(f"\n{'='*80}")
    print(f"Common valid samples across all sites: {len(common_valid_indices)}")
    print(f"{'='*80}")
    
    if len(common_valid_indices) == 0:
        print("\n✗ ERROR: No samples remain after filtering!")
        sys.exit(1)
    
    # Filter to common indices
    for i, feature_dict in enumerate(all_feature_dicts):
        site_valid_idx = feature_dict['valid_indices']
        mask = np.isin(site_valid_idx, common_valid_indices)
        
        for key in ['lig_inter', 'mut_inter', 'inter_interaction', 
                    'lig_intra', 'mut_intra', 'intra_interaction', 
                    'lig_mut_mix_inter_intra', 'final_fp_interaction']:
            all_feature_dicts[i][key] = feature_dict[key][mask]

    # Get y_train for activity (IC50/Ki values)
    y_train1 = activity_values_valid.iloc[common_valid_indices].values
    y_train1 = np.log1p(y_train1)

    y_scaler1 = StandardScaler()
    y_train_scaled1 = y_scaler1.fit_transform(y_train1.reshape(-1, 1)).flatten()

    # Get y_train2 for docking scores
    y_train2 = activity_values2_valid[common_valid_indices]

    y_scaler2 = StandardScaler()
    y_train_scaled2 = y_scaler2.fit_transform(y_train2.reshape(-1, 1)).flatten()

    # ===== STAGE 2: TRAIN HIERARCHICAL MODELS & EXTRACT EMBEDDINGS =====
    print("\n" + "="*80)
    print("STAGE 2: TRAIN HIERARCHICAL MODELS & EXTRACT EMBEDDINGS")
    print("="*80)
    
    all_scalers = []
    all_embeddings = []
    
    for site_idx, (site_name, _) in enumerate(mutation_sites):
        print(f"\nSite {site_idx+1}/8: {site_name}")
        feature_dict = all_feature_dicts[site_idx]
        
        scalers = {}
        scaled_features = {}
        for key in ['lig_inter', 'mut_inter', 'inter_interaction', 
                    'lig_intra', 'mut_intra', 'intra_interaction', 
                    'lig_mut_mix_inter_intra', 'final_fp_interaction']:
            scalers[key] = StandardScaler()
            scaled_features[key] = scalers[key].fit_transform(feature_dict[key])
        
        all_scalers.append(scalers)
        
        feature_dims = {k: v.shape[1] for k, v in scaled_features.items()}
        model = build_priority_hierarchical_model(feature_dims)

        checkpoint = ModelCheckpoint(f'hierarchical_model_{site_name}.h5', monitor='val_loss', save_best_only=True, verbose=0)
        early_stop = EarlyStopping(monitor='val_loss', patience=30, restore_best_weights=True, verbose=0)
        
        model.fit(
            x=[scaled_features[k] for k in ['final_fp_interaction', 'lig_mut_mix_inter_intra', 'inter_interaction', 
                                            'intra_interaction', 'mut_inter', 'lig_inter', 'mut_intra', 'lig_intra']],
            y={'activity_output': y_train_scaled1, 'docking_output': y_train_scaled2},
            epochs=100, batch_size=32, validation_split=0.2, callbacks=[early_stop, checkpoint], verbose=1
        )
        
        embedding_model = Model(inputs=model.inputs, outputs=model.get_layer('embedding_output').output)
        embeddings = embedding_model.predict([scaled_features[k] for k in ['final_fp_interaction', 'lig_mut_mix_inter_intra', 'inter_interaction', 
                                                                           'intra_interaction', 'mut_inter', 'lig_inter', 'mut_intra', 'lig_intra']], verbose=0)
        all_embeddings.append(embeddings)
    
    # ===== STAGE 3: TRAIN RNN-LSTM-KAN MODEL =====
    print("\n" + "="*80)
    print("STAGE 3: TRAIN RNN-LSTM-KAN SEQUENTIAL MODEL")
    print("="*80)
    
    sequential_embeddings = np.stack(all_embeddings, axis=1)
    rnn_model = build_rnn_sequential_model(sequential_embeddings.shape[2], sequential_embeddings.shape[1])

    rnn_checkpoint = ModelCheckpoint('rnn_sequential_model.h5', monitor='val_loss', save_best_only=True, verbose=0)
    rnn_early_stop = EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True, verbose=0)
    
    rnn_history = rnn_model.fit(
        x=sequential_embeddings,
        y={'final_activity_output': y_train_scaled1, 'final_docking_output': y_train_scaled2},
        epochs=150, batch_size=32, validation_split=0.2, callbacks=[rnn_early_stop, rnn_checkpoint], verbose=1
    )

    

    # ===== SAVE SCALERS =====
    with open('feature_scalers.pkl', 'wb') as f: pickle.dump(all_scalers, f)
    with open('y_scalers.pkl', 'wb') as f: pickle.dump({'y_scaler1': y_scaler1, 'y_scaler2': y_scaler2}, f)
    unique_mutation_profiles.to_csv('mutation_profiles.csv', index=False)

   
    # ===== PLOT TRAINING HISTORY =====
    print("\n" + "="*80)
    print("PLOTTING TRAINING HISTORY")
    print("="*80)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss plot
    axes[0].plot(rnn_history.history['loss'], label='Train Loss', linewidth=2, color='#2E86AB')
    axes[0].plot(rnn_history.history['val_loss'], label='Val Loss', linewidth=2, color='#A23B72')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title('RNN-LSTM-KAN Model - Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # MAE plot
    axes[1].plot(rnn_history.history['final_activity_output_mae'], label='Train Activity MAE', linewidth=2, color='#2E86AB')
    axes[1].plot(rnn_history.history['val_final_activity_output_mae'], label='Val Activity MAE', linewidth=2, color='#A23B72')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].set_title('RNN-LSTM-KAN Model - MAE', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kan_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Training history plot saved: kan_training_history.png")

if __name__ == "__main__":
    main()

#TODO
# 1. Find the most suiiable KAN layer 
