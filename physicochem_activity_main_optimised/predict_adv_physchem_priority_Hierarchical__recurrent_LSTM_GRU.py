#!/usr/bin/env python3


import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr, spearmanr
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate, Multiply, LSTM, GRU, Bidirectional, LeakyReLU
from tensorflow.keras.optimizers import Adam # Needed for recompile
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, MolSurf, GraphDescriptors, Fragments
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit import DataStructs
from numpy.linalg import norm
from rdkit import RDLogger

# Disable RDKit warnings
RDLogger.DisableLog('rdApp.*')

# ============================================================================
# FEATURE GENERATION FUNCTIONS (
# ============================================================================

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
        features.append(MolSurf.TPSA(mol))#The sum of the surface areas of all polar atoms (mostly oxygen and nitrogen) and their attached hydrogens.High TPSA â†’ more polar, less membrane permeable, more soluble, Low TPSA â†’ less polar, more membrane permeable (good for oral drugs)
        
        features.append(MolSurf.LabuteASA(mol))#An approximation of the total solvent-accessible surface area (SASA) of the molecule, calculated using Labuteâ€™s algorithm. #Reflects molecular size and hydrophobic surface exposure
        #Aiâ€‹=Siâ€‹â‹…Piâ€‹, S = 4Ï€r^2i , Piâ€‹=1âˆ’jâˆ â€‹(1âˆ’fij
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


def generate_final_interaction_features(lig_smiles, mut_smiles): # fingerprints, morgan fingerprints dominate
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

    # assuming got positive charges ligand and negative charge mut with weighted size molwt
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

def generate_all_features(lig_smiles, mut_smiles, ligand_cache=None, mutation_cache=None, interaction_cache=None):
    """
    Generate all feature sets for a given ligand-mutation pair.
    Returns a dictionary matching the keys used in prediction loop.

    Optional cache dicts (matching the caching strategy used in the KAN predict script):
      - ligand_cache:      keyed by ligand SMILES      -> (lig_inter, lig_intra)
      - mutation_cache:    keyed by mutation-site SMILES -> (mut_inter, mut_intra)
      - interaction_cache: keyed by (lig_smiles, mut_smiles) -> (lig_mut_mix_inter_intra,
                            inter_interaction, intra_interaction, final_fp_interaction)

    This function is called once per (row, site) pair in the prediction loop. The ligand
    SMILES is identical across all 8 mechanistic sites within a row, and the mutation-site
    SMILES is identical across every row that shares the same tkd class (only 8 classes
    total) -- so without caching, the same ligand/mutation descriptors get recomputed many
    times over. Passing cache dicts in avoids that; passing none (the default) reproduces
    the original uncached behaviour.
    """
    if ligand_cache is None:
        ligand_cache = {}
    if mutation_cache is None:
        mutation_cache = {}
    if interaction_cache is None:
        interaction_cache = {}

    # 1. Generate individual features (cached per unique SMILES string)
    if lig_smiles in ligand_cache:
        lig_inter, lig_intra = ligand_cache[lig_smiles]
    else:
        lig_inter = generate_lig_inter_features(lig_smiles)
        lig_intra = generate_lig_intra_features(lig_smiles)
        ligand_cache[lig_smiles] = (lig_inter, lig_intra)

    if mut_smiles in mutation_cache:
        mut_inter, mut_intra = mutation_cache[mut_smiles]
    else:
        mut_inter = generate_mut_inter_features(mut_smiles)
        mut_intra = generate_mut_intra_features(mut_smiles)
        mutation_cache[mut_smiles] = (mut_inter, mut_intra)

    if any(x is None for x in [lig_inter, lig_intra, mut_inter, mut_intra]):
        return None

    # 2. Generate interaction features (cached per unique ligand-mutation SMILES pair)
    pair_key = (lig_smiles, mut_smiles)

    if pair_key in interaction_cache:
        lig_mut_mix_inter_intra, inter_interaction, intra_interaction, final_fp_interaction = interaction_cache[pair_key]
    else:
        # Custom features
        lig_mut_inter, lig_mut_intra, lig_mut_mix_inter_intra = generate_custom_features(
            lig_inter, mut_inter, lig_intra, mut_intra
        )

        # Similarity features
        inter_interaction = generate_inter_interaction_features(lig_inter, mut_inter)
        intra_interaction = generate_intra_interaction_features(lig_intra, mut_intra)

        # 3. Concatenate custom features into interaction vectors
        # (Matches logic in generate_hierarchical_features)
        if len(lig_mut_inter) > 0:
            inter_interaction = np.concatenate([np.array(lig_mut_inter), inter_interaction])

        if len(lig_mut_intra) > 0:
            intra_interaction = np.concatenate([np.array(lig_mut_intra), intra_interaction])

        # 4. Fingerprint features
        final_fp_interaction = generate_final_interaction_features(lig_smiles, mut_smiles)

        interaction_cache[pair_key] = (lig_mut_mix_inter_intra, inter_interaction, intra_interaction, final_fp_interaction)

    return {
        'lig_inter': lig_inter,
        'mut_inter': mut_inter,
        'inter_interaction': inter_interaction,
        'lig_intra': lig_intra,
        'mut_intra': mut_intra,
        'intra_interaction': intra_interaction,
        'lig_mut_mix_inter_intra': np.array(lig_mut_mix_inter_intra),
        'final_fp_interaction': final_fp_interaction
    }


print("=" * 80)
print("PREDICTION SCRIPT FOR ADV_PHYSCHEM5F2")
print("Hierarchical 8-site RNN Model")
print("=" * 80)

# Mechanistic reaction order for EGFR: full -> ATP pocket -> P-loop -> C-helix ->
# 19-deletions -> hinge loop (T790M/C797S) -> A-loop DFG -> HRD catalytic motif
# This order MUST match the training script exactly, since it defines the RNN timestep order.
MUTATION_SITES = ['FULL_SMILES', 'ATP_POCKET', 'P_LOOP', 'C_HELIX', 'DEL19', 'HINGE_LOOP', 'DFG_A_LOOP', 'HRD_CAT']
SITE_COLUMNS = [
    'smiles_full_sequence_egfr_manual',
    'smiles_sequence_atp_ pocket',
    'smiles_sequence_p_loop_constant',
    'smiles_sequence_c_helix_constant',
    'smiles_sequence_19_deletions',
    'smiles_sequence_hinge_loop_t790m_c797s',
    'smiles_sequence_a_loop_dfg',
    'smiles_sequence_hrd_constant'
]


def load_models_and_scalers(model_dir):
    """Load all required models and scalers"""
    
    print("\nLoading models and scalers...")
    
    # Load hierarchical models for each site
    hierarchical_models = {}
    for site_name in MUTATION_SITES:
        model_path = os.path.join(model_dir, f'hierarchical_model_{site_name}.h5')
        if os.path.exists(model_path):
            hierarchical_models[site_name] = load_model(model_path, compile=False)
            print(f"  ✓ Loaded {site_name} model")
        else:
            raise FileNotFoundError(f"Model not found: {model_path}")
    
    # Load RNN model
    rnn_path = os.path.join(model_dir, 'rnn_sequential_model.h5')
    if os.path.exists(rnn_path):
        rnn_model = load_model(rnn_path, compile=False)
        print(f"  ✓ Loaded RNN model")
    else:
        raise FileNotFoundError(f"RNN model not found: {rnn_path}")
    
    # Load scalers
    # Load feature scalers
    with open(os.path.join(model_dir, 'feature_scalers.pkl'), 'rb') as f:
        all_scalers = pickle.load(f)
    print(f"  ✓ Loaded feature scalers")
    
    # Load y scalers
    with open(os.path.join(model_dir, 'y_scalers.pkl'), 'rb') as f:
        y_scalers = pickle.load(f)
    y_scaler1 = y_scalers['y_scaler1']
    y_scaler2 = y_scalers['y_scaler2']
    print(f"  ✓ Loaded y scalers")
    
    return hierarchical_models, rnn_model, all_scalers, y_scaler1, y_scaler2

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
    Make predictions using the trained hierarchical RNN model.
    
    Parameters:
    -----------
    input_csv : str
        Path to input CSV file with 'smiles' and 'tkd' columns
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
    
    # Check required columns
    required_cols = ['smiles', 'tkd']
    if not all(col in df_pred.columns for col in required_cols):
        raise ValueError(f"Input CSV must contain 'smiles' and 'tkd' columns. Found: {df_pred.columns.tolist()}")
    
    # Dataset fix: normalize whitespace in 'tkd' (e.g. a stray leading-space variant of
    # 'l858r/t790m/c797s triple' would otherwise be treated as an unseen mutation class).
    df_pred['tkd'] = df_pred['tkd'].astype(str).str.strip()
    
    # Check if ground truth exists
    has_ground_truth = 'standard value' in df_pred.columns and 'dock' in df_pred.columns
    
    # Load Model Files and Scalers
    print("Loading model files...")
    try:
        hierarchical_models, rnn_model, all_scalers, y_scaler1, y_scaler2 = load_models_and_scalers(model_dir)
        
        # Load mutation profiles
        mutation_profiles_path = os.path.join(model_dir, 'mutation_profiles.csv')
        if not os.path.exists(mutation_profiles_path):
            raise FileNotFoundError(f"Mutation profiles not found: {mutation_profiles_path}")
        
        df_mutation_profiles = pd.read_csv(mutation_profiles_path)
        df_mutation_profiles.columns = df_mutation_profiles.columns.str.strip()
        df_mutation_profiles['tkd'] = df_mutation_profiles['tkd'].astype(str).str.strip()
        print(f"  ✓ Loaded {len(df_mutation_profiles)} mutation profiles")
        
    except FileNotFoundError as e:
        print(f"Error loading model files: {e}")
        print("Please ensure all model files are in the model directory:")
        print("  - hierarchical_model_*.h5 (8 files)")
        print("  - rnn_sequential_model.h5")
        print("  - feature_scalers.pkl")
        print("  - y_scalers.pkl")
        print("  - mutation_profiles.csv")
        return None

    print(f"\nTotal prediction samples: {len(df_pred)}")
    
    # Recompile hierarchical models
    print("\nRecompiling hierarchical models...")
    for site_name, model in hierarchical_models.items():
        model.compile(
            optimizer=Adam(learning_rate=0.003),
            loss={
                'activity_output': 'mean_squared_error',
                'docking_output': 'mean_squared_error'
            },
            loss_weights={
                'activity_output': 1.0,
                'docking_output': 0.7
            },
            metrics={
                'activity_output': ['mae', 'mse'],
                'docking_output': ['mae', 'mse']
            }
        )
    
    # Recompile RNN model
    rnn_model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={
            'final_activity_output': 'mean_squared_error',
            'final_docking_output': 'mean_squared_error'
        },
        loss_weights={
            'final_activity_output': 1.0,
            'final_docking_output': 0.5
        },
        metrics={
            'final_activity_output': ['mae', 'mse'],
            'final_docking_output': ['mae', 'mse']
        }
    )
    
    # Process predictions
    results = []

    # Feature caches, shared across every (row, site) pair processed below.
    # Matches the caching strategy used in the KAN predict script: avoids recomputing
    # ligand descriptors (identical across all 8 sites within a row), mutation-site
    # descriptors (identical across every row sharing the same tkd class -- only 8
    # classes total), and their pairwise interaction features.
    ligand_cache = {}
    mutation_cache = {}
    interaction_cache = {}

    print("\nProcessing predictions...")
    for pred_idx, pred_row in df_pred.iterrows():
        lig_smiles = pred_row['smiles']
        mutant_name = pred_row['tkd']
        
        # Validate SMILES
        if pd.isna(lig_smiles) or lig_smiles == '':
            print(f"Warning: Empty SMILES at row {pred_idx}, skipping")
            continue

        # Validate Mutation
        if pd.isna(mutant_name):
            print(f"Warning: Empty mutation at row {pred_idx}, skipping")
            continue
        
        # Find mutation profile
        mutation_profile = df_mutation_profiles[df_mutation_profiles['tkd'] == mutant_name]
        
        if len(mutation_profile) == 0:
            print(f"Warning: Mutation '{mutant_name}' not found in training data, skipping")
            continue
        
        mutation_profile = mutation_profile.iloc[0]
        
        # Get mutation SMILES for all sites
        # Mechanistic reaction order for EGFR: full -> ATP pocket -> P-loop -> C-helix ->
        # 19-deletions -> hinge loop (T790M/C797S) -> A-loop DFG -> HRD catalytic motif
        full_smiles = mutation_profile['smiles_full_sequence_egfr_manual']
        mut_atp = mutation_profile['smiles_sequence_atp_ pocket']
        mut_p_loop = mutation_profile['smiles_sequence_p_loop_constant']
        mut_c_helix = mutation_profile['smiles_sequence_c_helix_constant']
        mut_19_del = mutation_profile['smiles_sequence_19_deletions']
        mut_hinge_loop = mutation_profile['smiles_sequence_hinge_loop_t790m_c797s']
        mut_dfg_a_loop = mutation_profile['smiles_sequence_a_loop_dfg']
        mut_hrd_cat = mutation_profile['smiles_sequence_hrd_constant']
        
        # Process through each hierarchical site
        embeddings_all_sites = []
        valid_site_processing = True
        
        for site_idx, (site_name, mutation_smiles) in enumerate([
            ('FULL_SMILES', full_smiles),
            ('ATP_POCKET', mut_atp),
            ('P_LOOP', mut_p_loop),
            ('C_HELIX', mut_c_helix),
            ('DEL19', mut_19_del),
            ('HINGE_LOOP', mut_hinge_loop),
            ('DFG_A_LOOP', mut_dfg_a_loop),
            ('HRD_CAT', mut_hrd_cat)
        ]):
            
            # Generate features using the correct function (with caching)
            feature_dict = generate_all_features(
                lig_smiles, mutation_smiles,
                ligand_cache=ligand_cache,
                mutation_cache=mutation_cache,
                interaction_cache=interaction_cache
            )
            
            if feature_dict is None:
                print(f"Warning: Could not generate features for {mutant_name} at {site_name}, skipping this compound")
                valid_site_processing = False
                break
            
            # Scale features
            scalers = all_scalers[site_idx]
            scaled_features = {
                'final_fp_interaction': scalers['final_fp_interaction'].transform(feature_dict['final_fp_interaction'].reshape(1, -1)),
                'lig_mut_mix_inter_intra': scalers['lig_mut_mix_inter_intra'].transform(feature_dict['lig_mut_mix_inter_intra'].reshape(1, -1)),
                'inter_interaction': scalers['inter_interaction'].transform(feature_dict['inter_interaction'].reshape(1, -1)),
                'intra_interaction': scalers['intra_interaction'].transform(feature_dict['intra_interaction'].reshape(1, -1)),
                'mut_inter': scalers['mut_inter'].transform(feature_dict['mut_inter'].reshape(1, -1)),
                'lig_inter': scalers['lig_inter'].transform(feature_dict['lig_inter'].reshape(1, -1)),
                'mut_intra': scalers['mut_intra'].transform(feature_dict['mut_intra'].reshape(1, -1)),
                'lig_intra': scalers['lig_intra'].transform(feature_dict['lig_intra'].reshape(1, -1))
            }
            
            # Get embedding from hierarchical model
            hierarchical_model = hierarchical_models[site_name]
            embedding_model = Model(
                inputs=hierarchical_model.inputs,
                outputs=hierarchical_model.get_layer('embedding_output').output
            )
            
            site_embedding = embedding_model.predict([
                scaled_features['final_fp_interaction'],
                scaled_features['lig_mut_mix_inter_intra'],
                scaled_features['inter_interaction'],
                scaled_features['intra_interaction'],
                scaled_features['mut_inter'],
                scaled_features['lig_inter'],
                scaled_features['mut_intra'],
                scaled_features['lig_intra']
            ], verbose=0)
            
            embeddings_all_sites.append(site_embedding)
        
        if not valid_site_processing:
            continue
        
        # Stack embeddings for RNN
        sequential_input = np.stack(embeddings_all_sites, axis=1)
        
        # Get final predictions from RNN
        predictions = rnn_model.predict(sequential_input, verbose=0)
        pred_activity_scaled = predictions[0].flatten()[0]
        pred_docking_scaled = predictions[1].flatten()[0]
        
        # Inverse transform
        pred_activity_log1p = y_scaler1.inverse_transform([[pred_activity_scaled]])[0, 0]
        pred_activity = np.expm1(pred_activity_log1p)
        pred_docking = y_scaler2.inverse_transform([[pred_docking_scaled]])[0, 0]
        
        # Store result
        res = {
            'smiles': lig_smiles,
            'mutation': mutant_name,
            'predicted_activity': pred_activity,
            'predicted_docking': pred_docking
        }
        
        if has_ground_truth:
            res['actual_activity'] = pred_row['standard value']
            res['actual_docking'] = pred_row['dock']
            
        # Keep other columns (including tkd for evaluation)
        for col in df_pred.columns:
            if col not in res and col not in ['smiles', 'standard value', 'dock']:
                 res[col] = pred_row[col]
                 
        results.append(res)
        
        if (pred_idx + 1) % 10 == 0:
            print(f"  Processed {pred_idx + 1}/{len(df_pred)} samples")
    
    if not results:
        print("Error: No valid samples found to predict.")
        return None

    print(f"\nFeature cache summary:")
    print(f"  Unique ligand SMILES seen:        {len(ligand_cache)}")
    print(f"  Unique mutation-site SMILES seen: {len(mutation_cache)}")
    print(f"  Unique ligand-mutation pairs seen: {len(interaction_cache)}")

    df_results = pd.DataFrame(results)
    
    # Save output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, 'predictions_hierarchical_lstm_gru.csv')
    df_results.to_csv(output_path, index=False)
    print(f"\n✓ Predictions saved to: {output_path}")
    
    # Evaluation (if ground truth exists)
    if has_ground_truth and len(df_results) > 0:
        evaluate_and_plot(df_results, output_dir, 'hierarchical_lstm_gru')
        
    return df_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Predictions')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    parser.add_argument('--model_dir', type=str, default='.', help='Model directory')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory')
    
    args = parser.parse_args()
    
    results = make_predictions(args.input, args.model_dir, args.output_dir)
    
    print(f"\n✓ Complete! Total predictions: {len(results)}")
    print(f"✓ Mutations covered: {results['mutation'].nunique()}")

    #python predict_adv_physchem_priority_Hierarchical__recurrent_LSTM_GRU.py --input validated_july_2026_testset_valid_tki.csv --model_dir . --output_dir ./prediction_output