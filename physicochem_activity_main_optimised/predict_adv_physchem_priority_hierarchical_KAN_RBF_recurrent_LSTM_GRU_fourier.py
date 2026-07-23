#!/usr/bin/env python3


import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr
# TensorFlow/Keras imports
os.environ["TF_USE_LEGACY_KERAS"] = "1"
import tensorflow as tf
from tensorflow.keras.models import load_model, Model

# RDKit imports
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, MolSurf, GraphDescriptors, Fragments
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit import DataStructs
from numpy.linalg import norm
from loguru import logger

# Logger configuration
logger.remove()
logger.add(sys.stderr, level="INFO")

#Division for custom interaction features using intermolecular and intramolecular forces
def safe_divide(numerator, denominator, default=0.0):
    """Safe division with default value for zero denominator"""
    if isinstance(denominator, (int, float)):
        return numerator / denominator if denominator != 0 else default
    else:
        result = np.where(denominator != 0, numerator / denominator, default)
        return result


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


print("=" * 80)
print("PREDICTION SCRIPT FOR KAN FOURIER MODEL")
print("Hierarchical 8-site RNN with Gaussian RBF and Fourier KAN layers")
print("=" * 80)

# ============================================================================
# KAN LAYER DEFINITIONS
# ============================================================================

class KANLayer(tf.keras.layers.Layer):
    """KAN Layer with Gaussian RBF basis functions"""
    
    def __init__(self, out_features, grid_size=20, grid_range=[-2.0, 2.0], **kwargs):
        super(KANLayer, self).__init__(**kwargs)
        self.out_features = out_features
        self.grid_size = grid_size
        self.grid_min, self.grid_max = grid_range
        
    def build(self, input_shape):
        in_features = input_shape[-1]
        
        # Grid initialization
        self.grid = tf.linspace(self.grid_min, self.grid_max, self.grid_size)
        self.grid = tf.cast(self.grid, dtype=tf.float32)
        
        self.mu = self.add_weight(
            name='mu',
            shape=(self.grid_size,),
            initializer=tf.keras.initializers.Constant(self.grid.numpy()),
            trainable=False 
        )
        
        # Sigma (bandwidth)
        spacing = (self.grid_max - self.grid_min) / (self.grid_size - 1)
        self.sigma = spacing
        
        # Base Weights
        self.base_weight = self.add_weight(
            name='base_weight',
            shape=(in_features, self.out_features),
            initializer='glorot_uniform',
            trainable=True
        )
        
        # Spline Weights (RBF coefficients)
        self.spline_weight = self.add_weight(
            name='spline_weight',
            shape=(in_features, self.grid_size, self.out_features),
            initializer='glorot_uniform',
            trainable=True
        )
        
        super(KANLayer, self).build(input_shape)

    def call(self, x):
        # Base Feature Transformation (SiLU activation)
        base = tf.nn.silu(x)
        base_out = tf.einsum('...i,io->...o', base, self.base_weight)
        
        # Spline Part (RBF Expansion)
        x_expanded = tf.expand_dims(x, -1)
        diff = x_expanded - self.mu
        
        # Gaussian RBF basis functions
        basis = tf.exp(-tf.math.pow(diff / self.sigma, 2))
        
        # Compute spline output
        spline_out = tf.einsum('...ig,igo->...o', basis, self.spline_weight)
        
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


class FourierKANLayer(tf.keras.layers.Layer):
    """Fourier KAN Layer with sinusoidal basis functions"""
    
    def __init__(self, out_features, grid_size=5, add_bias=True, domain="[-pi, pi]", **kwargs):
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
# NOTE: an earlier version of this script had a dead, unused
# load_models_and_scalers() helper here whose MUTATION_SITES/SITE_COLUMNS
# constants used inconsistent site-name labels and stale column names
# (never actually called anywhere in this file). It has been removed;
# the real model/scaler loading + mutation-site handling used at prediction
# time lives in make_predictions() below, which is what actually executes.
# ============================================================================


# ============================================================================
# UPDATED EVALUATION AND PLOTTING FUNCTION
# ============================================================================

def evaluate_and_plot(df_results, output_dir, model_name):
    """
    Evaluate predictions and generate comprehensive plots with statistics
    
    Updates:
    - Save both Pearson and Spearman metrics to CSV table
    - Create individual plots for each mutation
    - Generate correlation plots per mutation (both Pearson and Spearman)
    - Save all plots separately
    """
    
    print("\n" + "=" * 80)
    print("EVALUATION METRICS")
    print("=" * 80)
    
    # Create metrics directory
    metrics_dir = os.path.join(output_dir, 'metrics')
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(metrics_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # ========================================================================
    # 1. OVERALL METRICS
    # ========================================================================
    mae_act = mean_absolute_error(df_results['actual_activity'], df_results['predicted_activity'])
    rmse_act = np.sqrt(mean_squared_error(df_results['actual_activity'], df_results['predicted_activity']))
    mae_dock = mean_absolute_error(df_results['actual_docking'], df_results['predicted_docking'])
    rmse_dock = np.sqrt(mean_squared_error(df_results['actual_docking'], df_results['predicted_docking']))
    
    # Calculate correlations for overall
    pearson_act, pval_act_p = pearsonr(df_results['actual_activity'], df_results['predicted_activity'])
    pearson_dock, pval_dock_p = pearsonr(df_results['actual_docking'], df_results['predicted_docking'])
    spearman_act, pval_act_s = spearmanr(df_results['actual_activity'], df_results['predicted_activity'])
    spearman_dock, pval_dock_s = spearmanr(df_results['actual_docking'], df_results['predicted_docking'])
    
    print(f"\nOverall Performance:")
    print(f"  Activity  - MAE: {mae_act:.4f}, RMSE: {rmse_act:.4f}, Pearson R: {pearson_act:.4f}, Spearman ρ: {spearman_act:.4f}")
    print(f"  Docking   - MAE: {mae_dock:.4f}, RMSE: {rmse_dock:.4f}, Pearson R: {pearson_dock:.4f}, Spearman ρ: {spearman_dock:.4f}")
    
    # ========================================================================
    # 2. PER-MUTATION METRICS
    # ========================================================================
    print("\n" + "=" * 80)
    print("PER-MUTATION METRICS")
    print("=" * 80)
    
    # Store metrics for CSV
    metrics_data = []
    
    # Add overall metrics first
    metrics_data.append({
        'Mutation': 'Overall',
        'N_Samples': len(df_results),
        'Activity_MAE': mae_act,
        'Activity_RMSE': rmse_act,
        'Activity_Pearson_R': pearson_act,
        'Activity_Pearson_pval': pval_act_p,
        'Activity_Spearman_rho': spearman_act,
        'Activity_Spearman_pval': pval_act_s,
        'Docking_MAE': mae_dock,
        'Docking_RMSE': rmse_dock,
        'Docking_Pearson_R': pearson_dock,
        'Docking_Pearson_pval': pval_dock_p,
        'Docking_Spearman_rho': spearman_dock,
        'Docking_Spearman_pval': pval_dock_s
    })
    
    # Process each mutation
    mutations = sorted(df_results['tkd'].unique())
    
    for mutation in mutations:
        mut_data = df_results[df_results['tkd'] == mutation]
        n_samples = len(mut_data)
        
        if n_samples < 2:
            print(f"\n{mutation}: Insufficient data (n={n_samples}), skipping")
            continue
        
        # Calculate metrics
        mae_a = mean_absolute_error(mut_data['actual_activity'], mut_data['predicted_activity'])
        rmse_a = np.sqrt(mean_squared_error(mut_data['actual_activity'], mut_data['predicted_activity']))
        mae_d = mean_absolute_error(mut_data['actual_docking'], mut_data['predicted_docking'])
        rmse_d = np.sqrt(mean_squared_error(mut_data['actual_docking'], mut_data['predicted_docking']))
        
        # Correlations
        try:
            pearson_a, pval_a_p = pearsonr(mut_data['actual_activity'], mut_data['predicted_activity'])
            pearson_d, pval_d_p = pearsonr(mut_data['actual_docking'], mut_data['predicted_docking'])
            spearman_a, pval_a_s = spearmanr(mut_data['actual_activity'], mut_data['predicted_activity'])
            spearman_d, pval_d_s = spearmanr(mut_data['actual_docking'], mut_data['predicted_docking'])
        except:
            pearson_a, pval_a_p = np.nan, np.nan
            pearson_d, pval_d_p = np.nan, np.nan
            spearman_a, pval_a_s = np.nan, np.nan
            spearman_d, pval_d_s = np.nan, np.nan
        
        print(f"\n{mutation} (n={n_samples}):")
        print(f"  Activity  - MAE: {mae_a:.4f}, RMSE: {rmse_a:.4f}, Pearson R: {pearson_a:.4f}, Spearman ρ: {spearman_a:.4f}")
        print(f"  Docking   - MAE: {mae_d:.4f}, RMSE: {rmse_d:.4f}, Pearson R: {pearson_d:.4f}, Spearman ρ: {spearman_d:.4f}")
        
        # Store in metrics list
        metrics_data.append({
            'Mutation': mutation,
            'N_Samples': n_samples,
            'Activity_MAE': mae_a,
            'Activity_RMSE': rmse_a,
            'Activity_Pearson_R': pearson_a,
            'Activity_Pearson_pval': pval_a_p,
            'Activity_Spearman_rho': spearman_a,
            'Activity_Spearman_pval': pval_a_s,
            'Docking_MAE': mae_d,
            'Docking_RMSE': rmse_d,
            'Docking_Pearson_R': pearson_d,
            'Docking_Pearson_pval': pval_d_p,
            'Docking_Spearman_rho': spearman_d,
            'Docking_Spearman_pval': pval_d_s
        })
    
    # ========================================================================
    # 3. SAVE METRICS TO CSV
    # ========================================================================
    metrics_df = pd.DataFrame(metrics_data)
    metrics_csv_path = os.path.join(metrics_dir, f'{model_name}_metrics_summary.csv')
    metrics_df.to_csv(metrics_csv_path, index=False, float_format='%.6f')
    print(f"\n✓ Metrics saved to: {metrics_csv_path}")
    
    # ========================================================================
    # 4. CREATE OVERALL COMBINED PLOT (2x2 layout)
    # ========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Activity scatter plot
    axes[0, 0].scatter(df_results['actual_activity'], df_results['predicted_activity'], 
                      alpha=0.5, s=20, edgecolors='k', linewidths=0.5)
    axes[0, 0].plot([df_results['actual_activity'].min(), df_results['actual_activity'].max()],
                   [df_results['actual_activity'].min(), df_results['actual_activity'].max()],
                   'r--', lw=2, label='Perfect prediction')
    axes[0, 0].set_xlabel('Actual Activity', fontsize=11)
    axes[0, 0].set_ylabel('Predicted Activity', fontsize=11)
    axes[0, 0].set_title(f'Activity (Overall)\nRMSE={rmse_act:.4f}, PR={pearson_act:.3f}, SR={spearman_act:.3f}', fontsize=12)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Docking scatter plot
    axes[0, 1].scatter(df_results['actual_docking'], df_results['predicted_docking'], 
                      alpha=0.5, s=20, edgecolors='k', linewidths=0.5)
    axes[0, 1].plot([df_results['actual_docking'].min(), df_results['actual_docking'].max()],
                   [df_results['actual_docking'].min(), df_results['actual_docking'].max()],
                   'r--', lw=2, label='Perfect prediction')
    axes[0, 1].set_xlabel('Actual Docking Score', fontsize=11)
    axes[0, 1].set_ylabel('Predicted Docking Score', fontsize=11)
    axes[0, 1].set_title(f'Docking (Overall)\nRMSE={rmse_dock:.4f}, PR={pearson_dock:.3f}, SR={spearman_dock:.3f}', fontsize=12)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Activity residuals by mutation
    mutation_colors = plt.cm.tab10(np.linspace(0, 1, len(mutations)))
    for idx, mutation in enumerate(mutations):
        mut_data = df_results[df_results['tkd'] == mutation]
        residuals = mut_data['actual_activity'] - mut_data['predicted_activity']
        axes[1, 0].scatter(mut_data['predicted_activity'], residuals, 
                         label=mutation, alpha=0.6, s=20, c=[mutation_colors[idx]])
    axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Predicted Activity', fontsize=11)
    axes[1, 0].set_ylabel('Residuals (Actual - Predicted)', fontsize=11)
    axes[1, 0].set_title('Activity Residuals by Mutation', fontsize=12)
    axes[1, 0].legend(fontsize='small', loc='best')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Docking residuals by mutation
    for idx, mutation in enumerate(mutations):
        mut_data = df_results[df_results['tkd'] == mutation]
        residuals = mut_data['actual_docking'] - mut_data['predicted_docking']
        axes[1, 1].scatter(mut_data['predicted_docking'], residuals, 
                         label=mutation, alpha=0.6, s=20, c=[mutation_colors[idx]])
    axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[1, 1].set_xlabel('Predicted Docking Score', fontsize=11)
    axes[1, 1].set_ylabel('Residuals (Actual - Predicted)', fontsize=11)
    axes[1, 1].set_title('Docking Residuals by Mutation', fontsize=12)
    axes[1, 1].legend(fontsize='small', loc='best')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    overall_plot_file = os.path.join(plots_dir, f'{model_name}_overall_combined.png')
    plt.savefig(overall_plot_file, dpi=300, bbox_inches='tight')
    print(f"✓ Overall combined plot saved to: {overall_plot_file}")
    plt.close()
    
    # ========================================================================
    # 5. CREATE INDIVIDUAL PLOTS FOR EACH MUTATION
    # ========================================================================
    print("\nGenerating individual mutation plots...")
    
    for mutation in mutations:
        mut_data = df_results[df_results['tkd'] == mutation]
        
        if len(mut_data) < 2:
            continue
        
        # Get metrics for this mutation
        mut_metrics = metrics_df[metrics_df['Mutation'] == mutation].iloc[0]
        
        # Create 2x2 subplot for each mutation
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle(f'Mutation: {mutation} (n={len(mut_data)})', fontsize=14, fontweight='bold')
        
        # Activity scatter
        axes[0, 0].scatter(mut_data['actual_activity'], mut_data['predicted_activity'], 
                          alpha=0.6, s=50, edgecolors='k', linewidths=0.8, c='steelblue')
        min_val = min(mut_data['actual_activity'].min(), mut_data['predicted_activity'].min())
        max_val = max(mut_data['actual_activity'].max(), mut_data['predicted_activity'].max())
        axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        axes[0, 0].set_xlabel('Actual Activity', fontsize=11)
        axes[0, 0].set_ylabel('Predicted Activity', fontsize=11)
        axes[0, 0].set_title(f'Activity Prediction\nMAE={mut_metrics["Activity_MAE"]:.4f}, RMSE={mut_metrics["Activity_RMSE"]:.4f}', fontsize=11)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Docking scatter
        axes[0, 1].scatter(mut_data['actual_docking'], mut_data['predicted_docking'], 
                          alpha=0.6, s=50, edgecolors='k', linewidths=0.8, c='darkorange')
        min_val = min(mut_data['actual_docking'].min(), mut_data['predicted_docking'].min())
        max_val = max(mut_data['actual_docking'].max(), mut_data['predicted_docking'].max())
        axes[0, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        axes[0, 1].set_xlabel('Actual Docking Score', fontsize=11)
        axes[0, 1].set_ylabel('Predicted Docking Score', fontsize=11)
        axes[0, 1].set_title(f'Docking Prediction\nMAE={mut_metrics["Docking_MAE"]:.4f}, RMSE={mut_metrics["Docking_RMSE"]:.4f}', fontsize=11)
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Activity residuals
        residuals_act = mut_data['actual_activity'] - mut_data['predicted_activity']
        axes[1, 0].scatter(mut_data['predicted_activity'], residuals_act, 
                          alpha=0.6, s=50, edgecolors='k', linewidths=0.8, c='steelblue')
        axes[1, 0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('Predicted Activity', fontsize=11)
        axes[1, 0].set_ylabel('Residuals (Actual - Predicted)', fontsize=11)
        axes[1, 0].set_title('Activity Residuals', fontsize=11)
        axes[1, 0].grid(True, alpha=0.3)
        
        # Docking residuals
        residuals_dock = mut_data['actual_docking'] - mut_data['predicted_docking']
        axes[1, 1].scatter(mut_data['predicted_docking'], residuals_dock, 
                          alpha=0.6, s=50, edgecolors='k', linewidths=0.8, c='darkorange')
        axes[1, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[1, 1].set_xlabel('Predicted Docking Score', fontsize=11)
        axes[1, 1].set_ylabel('Residuals (Actual - Predicted)', fontsize=11)
        axes[1, 1].set_title('Docking Residuals', fontsize=11)
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save mutation-specific plot
        mutation_safe = mutation.replace('/', '_').replace('\\\\', '_')
        mutation_plot_file = os.path.join(plots_dir, f'{model_name}_mutation_{mutation_safe}.png')
        plt.savefig(mutation_plot_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: {mutation_safe}.png")
        plt.close()
    
    # ========================================================================
    # 6. CREATE CORRELATION PLOTS FOR EACH MUTATION
    # ========================================================================
    print("\nGenerating correlation plots...")
    
    corr_dir = os.path.join(plots_dir, 'correlations')
    os.makedirs(corr_dir, exist_ok=True)
    
    for mutation in mutations:
        mut_data = df_results[df_results['tkd'] == mutation]
        
        if len(mut_data) < 2:
            continue
        
        # Get metrics for this mutation
        mut_metrics = metrics_df[metrics_df['Mutation'] == mutation].iloc[0]
        
        # Create figure with 1x2 subplots for activity and docking
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f'Correlations - {mutation} (n={len(mut_data)})', fontsize=14, fontweight='bold')
        
        # Activity correlation
        axes[0].scatter(mut_data['actual_activity'], mut_data['predicted_activity'], 
                       alpha=0.6, s=50, edgecolors='k', linewidths=0.8, c='steelblue')
        
        # Add regression line
        z = np.polyfit(mut_data['actual_activity'], mut_data['predicted_activity'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(mut_data['actual_activity'].min(), mut_data['actual_activity'].max(), 100)
        axes[0].plot(x_line, p(x_line), "g-", linewidth=2, label=f'Fit: y={z[0]:.3f}x+{z[1]:.3f}')
        
        # Perfect prediction line
        min_val = min(mut_data['actual_activity'].min(), mut_data['predicted_activity'].min())
        max_val = max(mut_data['actual_activity'].max(), mut_data['predicted_activity'].max())
        axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        
        axes[0].set_xlabel('Actual Activity', fontsize=12)
        axes[0].set_ylabel('Predicted Activity', fontsize=12)
        axes[0].set_title(f'Activity\nPearson R = {mut_metrics["Activity_Pearson_R"]:.3f}, Spearman ρ = {mut_metrics["Activity_Spearman_rho"]:.3f}', 
                         fontsize=11)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Docking correlation
        axes[1].scatter(mut_data['actual_docking'], mut_data['predicted_docking'], 
                       alpha=0.6, s=50, edgecolors='k', linewidths=0.8, c='darkorange')
        
        # Add regression line
        z = np.polyfit(mut_data['actual_docking'], mut_data['predicted_docking'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(mut_data['actual_docking'].min(), mut_data['actual_docking'].max(), 100)
        axes[1].plot(x_line, p(x_line), "g-", linewidth=2, label=f'Fit: y={z[0]:.3f}x+{z[1]:.3f}')
        
        # Perfect prediction line
        min_val = min(mut_data['actual_docking'].min(), mut_data['predicted_docking'].min())
        max_val = max(mut_data['actual_docking'].max(), mut_data['predicted_docking'].max())
        axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect prediction')
        
        axes[1].set_xlabel('Actual Docking Score', fontsize=12)
        axes[1].set_ylabel('Predicted Docking Score', fontsize=12)
        axes[1].set_title(f'Docking\nPearson R = {mut_metrics["Docking_Pearson_R"]:.3f}, Spearman ρ = {mut_metrics["Docking_Spearman_rho"]:.3f}', 
                         fontsize=11)
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save correlation plot
        mutation_safe = mutation.replace('/', '_').replace('\\\\', '_')
        corr_plot_file = os.path.join(corr_dir, f'{model_name}_correlations_{mutation_safe}.png')
        plt.savefig(corr_plot_file, dpi=300, bbox_inches='tight')
        print(f"  ✓ Saved: correlations_{mutation_safe}.png")
        plt.close()
    
    print("\n" + "=" * 80)
    print("EVALUATION COMPLETE")
    print("=" * 80)
    print(f"✓ Metrics CSV: {metrics_csv_path}")
    print(f"✓ Overall plot: {overall_plot_file}")
    print(f"✓ Individual mutation plots: {plots_dir}")
    print(f"✓ Correlation plots: {corr_dir}")
    print("=" * 80)

def make_predictions(input_csv, model_dir='.', output_dir='.'):
    """
    Make predictions using the trained RNN-LSTM-KAN hierarchical model.
    
    Parameters:
    -----------
    input_csv : str
        Path to input CSV file with 'smiles' and 'tkd' columns
        Optionally can include mutation site SMILES columns (mechanistic
        reaction order for EGFR: full -> ATP pocket -> P-loop -> C-helix ->
        19-deletions -> hinge loop (T790M/C797S) -> A-loop DFG -> HRD catalytic motif):
        - smiles_full_sequence_egfr_manual
        - smiles_sequence_atp_ pocket
        - smiles_sequence_p_loop_constant
        - smiles_sequence_c_helix_constant
        - smiles_sequence_19_deletions
        - smiles_sequence_hinge_loop_t790m_c797s
        - smiles_sequence_a_loop_dfg
        - smiles_sequence_hrd_constant
    model_dir : str
        Directory containing model files (default: current directory)
    output_dir : str
        Directory to save prediction outputs (default: current directory)
        
    Returns:
    --------
    df_results : pd.DataFrame
        DataFrame with predictions
    """
    
    print(f"Loading prediction data from: {input_csv}")
    df_pred = pd.read_csv(input_csv)
    df_pred.columns = df_pred.columns.str.strip()
    
    # Dataset fix: 'smiles_sequence_hinge_loop_t790m_c797s' can be saved twice (byte-identical
    # duplicate column) in the source data. Pandas auto-renames the 2nd occurrence to
    # '...t790m_c797s.1'; drop it if present.
    if 'smiles_sequence_hinge_loop_t790m_c797s.1' in df_pred.columns:
        df_pred = df_pred.drop(columns=['smiles_sequence_hinge_loop_t790m_c797s.1'])
    
    # Check required columns
    required_cols = ['smiles', 'tkd']
    if not all(col in df_pred.columns for col in required_cols):
        raise ValueError(f"Input CSV must contain 'smiles' and 'tkd' columns. Found: {df_pred.columns.tolist()}")
    
    # Dataset fix: normalize whitespace in 'tkd' (e.g. a stray leading-space variant of
    # 'l858r/t790m/c797s triple' would otherwise be treated as an unseen mutation class).
    df_pred['tkd'] = df_pred['tkd'].astype(str).str.strip()
    
    # Check if ground truth exists
    has_ground_truth = 'standard value' in df_pred.columns and 'dock' in df_pred.columns
    
    # Define mutation sites (must match training script exactly - this order defines the
    # RNN timestep order). Mechanistic reaction order for EGFR: full -> ATP pocket ->
    # P-loop -> C-helix -> 19-deletions -> hinge loop (T790M/C797S) -> A-loop DFG -> HRD motif
    mutation_sites = [
        ('FULL_SMILES', 'smiles_full_sequence_egfr_manual'),
        ('ATP_POCKET', 'smiles_sequence_atp_ pocket'),
        ('P_LOOP', 'smiles_sequence_p_loop_constant'),
        ('C_HELIX', 'smiles_sequence_c_helix_constant'),
        ('DEL19', 'smiles_sequence_19_deletions'),
        ('HINGE_LOOP', 'smiles_sequence_hinge_loop_t790m_c797s'),
        ('DFG_A_LOOP', 'smiles_sequence_a_loop_dfg'),
        ('HRD_CAT', 'smiles_sequence_hrd_constant')
    ]
    
    # Check if input CSV has mutation site columns
    mutation_site_cols = [col for _, col in mutation_sites]
    has_mutation_sites = all(col in df_pred.columns for col in mutation_site_cols)
    
    if has_mutation_sites:
        print("Using mutation site SMILES from input CSV")
        unique_mutation_profiles = df_pred[mutation_site_cols + ['tkd']].drop_duplicates(subset=['tkd']).reset_index(drop=True)
    else:
        print("Mutation site SMILES not in input CSV. Loading from saved profiles...")
        try:
            # Training script saves as CSV, not pickle
            unique_mutation_profiles = pd.read_csv(os.path.join(model_dir, 'mutation_profiles.csv'))
            unique_mutation_profiles.columns = unique_mutation_profiles.columns.str.strip()
            unique_mutation_profiles['tkd'] = unique_mutation_profiles['tkd'].astype(str).str.strip()
            print(f"Loaded {len(unique_mutation_profiles)} mutation profiles")
        except FileNotFoundError:
            print("\n" + "="*80)
            print("ERROR: Mutation site SMILES data not found!")
            print("="*80)
            print("\nYou have two options:")
            print("\n1. Include mutation site SMILES columns in your input CSV:")
            for _, col in mutation_sites:
                print(f"   - {col}")
            print("\n2. Ensure 'mutation_profiles.csv' exists in model directory")
            print("   (This is saved during training, after the RNN model finishes fitting)")
            print("="*80)
            return None
    
    # Load RNN model - correct filename from training script
    print("\nLoading RNN model...")
    try:
        rnn_model = load_model(
            os.path.join(model_dir, 'rnn_sequential_model.h5'),  # CORRECTED: was rnn_lstm_kan_model.h5
            custom_objects={'KANLayer': KANLayer, 'FourierKANLayer': FourierKANLayer},
            compile=False
        )
    except FileNotFoundError:
        print("Error: rnn_sequential_model.h5 not found in model directory.")
        return None
    
    # Load scalers - correct structure from training script
    print("Loading scalers...")
    try:
        # all_scalers is a list of dicts (one dict per site)
        with open(os.path.join(model_dir, 'feature_scalers.pkl'), 'rb') as f:
            all_scalers = pickle.load(f)
        
        # y_scalers is a dict with 'y_scaler1' and 'y_scaler2'
        with open(os.path.join(model_dir, 'y_scalers.pkl'), 'rb') as f:
            y_scalers = pickle.load(f)
            y_scaler1 = y_scalers['y_scaler1']
            y_scaler2 = y_scalers['y_scaler2']
    except FileNotFoundError as e:
        print(f"Error loading scalers: {e}")
        return None

    print(f"Total prediction samples: {len(df_pred)}")
    
    # Group by mutation
    all_results = []
    unique_mutations = df_pred['tkd'].unique()
    print(f"Found {len(unique_mutations)} unique mutations to process")
    
    for mutation_name in unique_mutations:
        print(f"\nProcessing mutation: {mutation_name}")
        mut_data = df_pred[df_pred['tkd'] == mutation_name]
        print(f"  Compounds: {len(mut_data)}")
        
        # Get mutation profile
        mut_profile = unique_mutation_profiles[unique_mutation_profiles['tkd'] == mutation_name]
        if len(mut_profile) == 0:
            print(f"  Warning: Mutation '{mutation_name}' not found in training data. Skipping.")
            continue
        
        mut_profile = mut_profile.iloc[0]
        
        # Extract mutation site SMILES
        mut_site_smiles = [mut_profile[site_col] for _, site_col in mutation_sites]
        
        # Validate mutation site SMILES
        if any(pd.isna(smi) or smi == '' for smi in mut_site_smiles):
            print(f"  Warning: Missing mutation site SMILES for '{mutation_name}'. Skipping.")
            continue
        
        # Generate embeddings for each site
        embeddings_all_sites = []
        valid_idx = None
        
        for site_idx, (site_name, _) in enumerate(mutation_sites):
            mut_smi_site = mut_site_smiles[site_idx]
            
            # Generate mutation features
            mut_inter = generate_mut_inter_features(mut_smi_site)
            mut_intra = generate_mut_intra_features(mut_smi_site)
            
            if mut_inter is None or mut_intra is None:
                print(f"  Warning: Could not generate mutation features for site {site_name}")
                break
            
            # Generate ligand features for all compounds
            site_features = {
                'lig_inter': [], 'mut_inter': [], 'inter_interaction': [],
                'lig_intra': [], 'mut_intra': [], 'intra_interaction': [],
                'lig_mut_mix_inter_intra': [], 'final_fp_interaction': []
            }
            site_valid_idx = []
            
            for idx, row in mut_data.iterrows():
                lig_smiles = row['smiles']
                
                if pd.isna(lig_smiles) or lig_smiles == '':
                    continue
                
                lig_inter = generate_lig_inter_features(lig_smiles)
                lig_intra = generate_lig_intra_features(lig_smiles)
                
                if lig_inter is None or lig_intra is None:
                    continue
                
                # Generate interaction features
                lig_mut_inter, lig_mut_intra, lig_mut_mix_inter_intra = generate_custom_features(
                    lig_inter, mut_inter, lig_intra, mut_intra
                )
                inter_interaction = generate_inter_interaction_features(lig_inter, mut_inter)
                intra_interaction = generate_intra_interaction_features(lig_intra, mut_intra)
                
                if len(lig_mut_inter) > 0:
                    inter_interaction = np.concatenate([np.array(lig_mut_inter), inter_interaction])
                if len(lig_mut_intra) > 0:
                    intra_interaction = np.concatenate([np.array(lig_mut_intra), intra_interaction])
                
                final_fp_interaction = generate_final_interaction_features(lig_smiles, mut_smi_site)
                
                site_features['lig_inter'].append(lig_inter)
                site_features['mut_inter'].append(mut_inter)
                site_features['inter_interaction'].append(inter_interaction)
                site_features['lig_intra'].append(lig_intra)
                site_features['mut_intra'].append(mut_intra)
                site_features['intra_interaction'].append(intra_interaction)
                site_features['lig_mut_mix_inter_intra'].append(np.array(lig_mut_mix_inter_intra))
                site_features['final_fp_interaction'].append(final_fp_interaction)
                site_valid_idx.append(idx)
            
            if len(site_valid_idx) == 0:
                print(f"  Warning: No valid features generated for site {site_name}")
                break
            
            if valid_idx is None:
                valid_idx = site_valid_idx
            
            # Scale features using the site-specific scalers
            scaled_features = {}
            scalers = all_scalers[site_idx]
            for key in site_features.keys():
                scaled_features[key] = scalers[key].transform(np.array(site_features[key]))
            
            # Load hierarchical model for this site
            try:
                h_model = load_model(
                    os.path.join(model_dir, f'hierarchical_model_{site_name}.h5'),
                    custom_objects={'KANLayer': KANLayer, 'FourierKANLayer': FourierKANLayer},
                    compile=False
                )
            except FileNotFoundError:
                print(f"  Error: Model file 'hierarchical_model_{site_name}.h5' not found")
                break
            
            # Get embeddings
            emb_model = Model(inputs=h_model.inputs, outputs=h_model.get_layer('embedding_output').output)
            site_embeddings = emb_model.predict(
                [scaled_features[k] for k in ['final_fp_interaction', 'lig_mut_mix_inter_intra', 
                                               'inter_interaction', 'intra_interaction', 
                                               'mut_inter', 'lig_inter', 'mut_intra', 'lig_intra']],
                verbose=0
            )
            embeddings_all_sites.append(site_embeddings)
        
        # Make predictions if all sites processed successfully
        if len(embeddings_all_sites) == len(mutation_sites) and valid_idx is not None:
            print(f"  Generating predictions for {len(valid_idx)} valid compounds...")
            sequential_embeddings = np.stack(embeddings_all_sites, axis=1)
            predictions = rnn_model.predict(sequential_embeddings, verbose=0)
            
            # Inverse transform predictions
            pred_activity = np.expm1(y_scaler1.inverse_transform(predictions[0].reshape(-1, 1)).flatten())
            pred_docking = y_scaler2.inverse_transform(predictions[1].reshape(-1, 1)).flatten()
            
            # Store results
            for i, idx in enumerate(valid_idx):
                row = df_pred.loc[idx]
                res = {
                    'smiles': row['smiles'],
                    'tkd': row['tkd'],
                    'predicted_activity': pred_activity[i],
                    'predicted_docking': pred_docking[i]
                }
                
                if has_ground_truth:
                    res['actual_activity'] = row['standard value']
                    res['actual_docking'] = row['dock']
                
                # Keep other columns
                for col in df_pred.columns:
                    if col not in res and col not in ['smiles', 'tkd', 'standard value', 'dock']:
                        res[col] = row[col]
                
                all_results.append(res)
            
            print(f"  ✓ Predicted {len(valid_idx)} compounds for {mutation_name}")
        else:
            print(f"  ✗ Failed to process mutation {mutation_name}")
    
    if not all_results:
        print("\n" + "="*80)
        print("ERROR: No valid predictions generated.")
        print("="*80)
        return None
    
    df_results = pd.DataFrame(all_results)
    
    # Save output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, 'predictions_rnn_lstm_kan.csv')
    df_results.to_csv(output_path, index=False)
    print(f"\n✓ Predictions saved to: {output_path}")
    print(f"✓ Total predictions: {len(df_results)}")
    print(f"✓ Mutations processed: {df_results['tkd'].nunique()}")
    
    # Evaluation (if ground truth exists)
    if has_ground_truth and len(df_results) > 0:
        evaluate_and_plot(df_results, output_dir, 'rnn_lstm_kan')
    
    return df_results
# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Predictions for KAN Fourier model')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    parser.add_argument('--model_dir', type=str, default='.', help='Model directory')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory')
    
    args = parser.parse_args()
    
    results = make_predictions(args.input, args.model_dir, args.output_dir)
    
    print(f"\n✓ Complete! Total predictions: {len(results)}")
    #print(f"✓ Mutations covered: {results['mutation'].nunique()}")

# to run script:  
#python predict_adv_physchem_priority_hierarchical_KAN_RBF_recurrent_LSTM_GRU_fourier.py --input validated_july_2026_testset_valid_tki.csv --model_dir . --output_dir ./prediction_output