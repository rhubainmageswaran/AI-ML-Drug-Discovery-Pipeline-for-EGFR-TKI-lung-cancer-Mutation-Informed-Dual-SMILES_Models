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
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Input, Concatenate, Multiply, LSTM, GRU, Bidirectional, LeakyReLU
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
print("RNN-LSTM INTEGRATED HIERARCHICAL MODEL FOR ADVANCED PHYSICOCHEMICAL DESCRIPTOR GENERATION")
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

#smiles for feature capture of ligand and mutation protein substructures
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

# unique_mutation_profiles = df_train_valid[mutation_profile_columns].drop_duplicates().reset_index(drop=True)
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
        features.append(MolSurf.TPSA(mol))#The sum of the surface areas of all polar atoms (mostly oxygen and nitrogen) and their attached hydrogens.High TPSA -> more polar, less membrane permeable, more soluble, Low TPSA -> less polar, more membrane permeable (good for oral drugs)
        
        features.append(MolSurf.LabuteASA(mol))#An approximation of the total solvent-accessible surface area (SASA) of the molecule, calculated using Labuteâ€™s algorithm. #Reflects molecular size and hydrophobic surface exposure
        #Aiâ€‹=Siâ€‹â‹…Piâ€‹, S = 4Ï€r^2i , Piâ€‹=1âˆ’jâˆâ€‹(1âˆ’fij
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
        return {'dice_sim': 0.0, 'tanimato': 0.0}
    
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, 2, nBits=2048)
    
    dice_sim = DataStructs.DiceSimilarity(fp1, fp2)
    tanimato = DataStructs.TanimotoSimilarity(fp1, fp2)

    #logger.info(f"Dice Similarity: {dice_sim}, Tanimoto Similarity: {tanimato}")

    
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
    # logger.info("inside generate_hierarchical_features function")
    # logger.info(f"Lig smiles sample: {ligand_smiles_series}, Mut smiles sample: {mutation_smiles_series}")
    # logger.debug(f"Input type={type(mutation_smiles_series)}, length={len(mutation_smiles_series)}, shape={(mutation_smiles_series.shape if hasattr(mutation_smiles_series, 'shape') else 'N/A')}")


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
        
        if idx == 0:  # Only print for first sample
            print(f"\n=== DEBUG: Sample 0 ===")
            print(f"Ligand SMILES: {lig_smi}")
            print(f"Mutation SMILES: {mut_smi}")

        # [MODIFIED] Check/Update Cache for Ligand Features
        if lig_smi in ligand_cache:
            lig_inter, lig_intra = ligand_cache[lig_smi]
        else:
            lig_inter = generate_lig_inter_features(lig_smi)
            lig_intra = generate_lig_intra_features(lig_smi)
            ligand_cache[lig_smi] = (lig_inter, lig_intra)

        # [MODIFIED] Check/Update Cache for Mutation Features
        if mut_smi in mutation_cache:
            mut_inter, mut_intra = mutation_cache[mut_smi]
        else:
            mut_inter = generate_mut_inter_features(mut_smi)
            mut_intra = generate_mut_intra_features(mut_smi)
            mutation_cache[mut_smi] = (mut_inter, mut_intra)

        if idx == 0:  # Only print for first sample
            print(f"lig_inter: {lig_inter}")
            print(f"mut_inter: {mut_inter}")
            print(f"lig_intra: {lig_intra}")
            print(f"mut_intra: {mut_intra}")
            print(f"=== END DEBUG ===\n")
        
        if any(f is None for f in [lig_inter, mut_inter, lig_intra, mut_intra]):
            continue
        
        # [MODIFIED] Check/Update Cache for Interaction Features (Dependent on Pair)
        pair_key = (lig_smi, mut_smi)
        
        if pair_key in interaction_cache:
            # Retrieve from cache
            lig_mut_mix_inter_intra, inter_interaction, intra_interaction, final_fp_interaction = interaction_cache[pair_key]
        else:
            #Generating custom representation features 
            lig_mut_inter, lig_mut_intra, lig_mut_mix_inter_intra = generate_custom_features(
                lig_inter, mut_inter, lig_intra, mut_intra
            )

            #Generating similarity features
            inter_interaction = generate_inter_interaction_features(lig_inter, mut_inter)
            intra_interaction = generate_intra_interaction_features(lig_intra, mut_intra)
            
            if len(lig_mut_inter) > 0:
                inter_interaction = np.concatenate([np.array(lig_mut_inter), inter_interaction])
            
            if len(lig_mut_intra) > 0:
                intra_interaction = np.concatenate([np.array(lig_mut_intra), intra_interaction])

            # Generating fingerprints
            final_fp_interaction = generate_final_interaction_features(lig_smi, mut_smi)
            
            # Save to cache
            interaction_cache[pair_key] = (lig_mut_mix_inter_intra, inter_interaction, intra_interaction, final_fp_interaction)
    

        # Single comprehensive logging before appending
        logger.info(f"Sample {idx} - Generated features: lig_inter shape={lig_inter.shape if hasattr(lig_inter, 'shape') else len(lig_inter)}, "
                   f"mut_inter shape={mut_inter.shape if hasattr(mut_inter, 'shape') else len(mut_inter)}, "
                   f"lig_intra shape={lig_intra.shape if hasattr(lig_intra, 'shape') else len(lig_intra)}, "
                   f"mut_intra shape={mut_intra.shape if hasattr(mut_intra, 'shape') else len(mut_intra)}, "
                   f"inter_interaction shape={inter_interaction.shape if hasattr(inter_interaction, 'shape') else len(inter_interaction)}, "
                   f"intra_interaction shape={intra_interaction.shape if hasattr(intra_interaction, 'shape') else len(intra_interaction)}, "
                   f"lig_mut_mix_inter_intra shape={len(lig_mut_mix_inter_intra)}, "
                   f"final_fp_interaction shape={final_fp_interaction.shape if hasattr(final_fp_interaction, 'shape') else len(final_fp_interaction)}")
        
        lig_inter_list.append(lig_inter)
        mut_inter_list.append(mut_inter)
        inter_interaction_list.append(inter_interaction)
        lig_intra_list.append(lig_intra)
        mut_intra_list.append(mut_intra)
        intra_interaction_list.append(intra_interaction)
        fp_interaction_list.append(final_fp_interaction)
        lig_mut_mix_inter_intra_list.append(np.array(lig_mut_mix_inter_intra))
        
        valid_indices.append(idx)
        
        # logger.info(f" lig_inter_list : {lig_inter_list}, mut_inter_list : {mut_inter_list}, inter_interaction_list : {inter_interaction_list}, lig_intra_list : {lig_intra_list}, mut_intra_list : {mut_intra_list}, intra_interaction_list : {intra_interaction_list}, fp_interaction_list : {fp_interaction_list}, lig_mut_mix_inter_intra_list : {lig_mut_mix_inter_intra_list}, valid_indices : {valid_indices}")
        # logger.debug(f" lig_inter_list len={len(lig_inter_list)}, mut_inter_list len={len(mut_inter_list)}, inter_interaction_list len={len(inter_interaction_list)}, lig_intra_list len={len(lig_intra_list)}, mut_intra_list len={len(mut_intra_list)}, intra_interaction_list len={len(intra_interaction_list)}, fp_interaction_list len={len(fp_interaction_list)}, lig_mut_mix_inter_intra_list len={len(lig_mut_mix_inter_intra_list)}, valid_indices len={len(valid_indices)}")
        
    print(f"  Successfully generated features for {len(valid_indices)} samples")
    
    feature_arrays = {
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
    
    return feature_arrays

#Forward Neural Network based on Priority Hierachy connecting and gating weights
#Higher-priority features control what lower-priority features contribute, creating a learned feature hierarchy rather than treating all inputs equally

#Rationale on Model:

#Concatenation to keep highest priority features as dominant baseline 
#Multiplification to filter and gate features on top
#LeakyReLU for hidden layers
#Tanh for embedding layers
#Sigmoid for gating mechanisms

def build_priority_hierarchical_model(feature_dims):

    logger.info("="*80)
    logger.info("BUILDING PRIORITY HIERARCHICAL MODEL")
    logger.info("="*80)
    
    # Log input dimensions
    logger.info("\n--- INPUT LAYER DIMENSIONS ---")
    for key, dim in feature_dims.items():
        logger.info(f"  {key}: ({dim},) -> Vector of length {dim}")
    logger.debug(f"Total input features: {feature_dims}")
    
    # Define inputs
    final_interaction_input = Input(shape=(feature_dims['final_fp_interaction'],), name='final_fp_interaction') #Fingerprint features
    lig_mut_mix_inter_intra_input = Input(shape=(feature_dims['lig_mut_mix_inter_intra'],), name='lig_mut_mix_inter_intra') #Custom features
    inter_interaction_input = Input(shape=(feature_dims['inter_interaction'],), name='inter_interaction') #Similarity features
    intra_interaction_input = Input(shape=(feature_dims['intra_interaction'],), name='intra_interaction') 
    mut_inter_input = Input(shape=(feature_dims['mut_inter'],), name='mut_inter') #Descriptor features
    lig_inter_input = Input(shape=(feature_dims['lig_inter'],), name='lig_inter')
    mut_intra_input = Input(shape=(feature_dims['mut_intra'],), name='mut_intra')
    lig_intra_input = Input(shape=(feature_dims['lig_intra'],), name='lig_intra')
    
    logger.info("\n" + "="*80)
    logger.info("PRIORITY 1: Final FP Interaction + Lig-Mut Mix Inter-Intra")
    logger.info("="*80)
    
    # Priority 1 - Final branch
    logger.info(f"\n[Final Branch]")
    logger.info(f"  Input: (batch, {feature_dims['final_fp_interaction']})")
    
    final_branch = Dense(32, kernel_initializer='he_normal', name='final_dense1')(final_interaction_input)
    logger.info(f"  Dense(32): (batch, {feature_dims['final_fp_interaction']}) @ (32, {feature_dims['final_fp_interaction']})^T -> (batch, 32)")
    logger.info(f"    Math: y = Wx + b, where W shape = ({feature_dims['final_fp_interaction']}, 32)")
    
    final_branch = LeakyReLU(alpha=0.1, name='final_leaky1')(final_branch)
    logger.info(f"  LeakyReLU(0.1): f(x) = x if x > 0 else 0.1*x -> (batch, 32)")
    
    final_branch = BatchNormalization(name='final_bn1')(final_branch)
    logger.info(f"  BatchNorm: normalize(x, mean, std) -> (batch, 32)")
    
    final_branch = Dropout(0.1, name='final_dropout1')(final_branch)
    logger.info(f"  Dropout(0.1): randomly zero 10% neurons during training -> (batch, 32)")
    
    final_branch = Dense(16, kernel_initializer='he_normal', name='final_dense2')(final_branch)
    logger.info(f"  Dense(16): (batch, 32) @ (16, 32)^T -> (batch, 16)")
    
    final_branch = LeakyReLU(alpha=0.1, name='final_leaky2')(final_branch)
    logger.info(f"  LeakyReLU(0.1): -> (batch, 16)")
    
    final_emb = Dense(8, activation='tanh', name='final_embedding')(final_branch)
    logger.info(f"  Dense(8, tanh): (batch, 16) @ (8, 16)^T -> (batch, 8)")
    logger.info(f"    tanh: f(x) = (e^x - e^-x)/(e^x + e^-x), range [-1, 1]")
    logger.info(f"  Final embedding output: (batch, 8)")
    
    # Priority 1 - Mix inter-intra branch 
    logger.info(f"\n[Lig-Mut Mix Inter-Intra Branch]")
    logger.info(f"  Input: (batch, {feature_dims['lig_mut_mix_inter_intra']})")
    
    lig_mut_mix_inter_intra_branch = Dense(8, kernel_initializer='he_normal', name='mix_inter_intra_dense')(lig_mut_mix_inter_intra_input)
    logger.info(f"  Dense(8): (batch, {feature_dims['lig_mut_mix_inter_intra']}) @ (8, {feature_dims['lig_mut_mix_inter_intra']})^T -> (batch, 8)")
    
    lig_mut_mix_inter_intra_branch = LeakyReLU(alpha=0.1, name='mix_inter_intra_leaky')(lig_mut_mix_inter_intra_branch)
    logger.info(f"  LeakyReLU(0.1): -> (batch, 8)")
    
    lig_mut_mix_inter_intra_branch_emb = Dense(4, activation='tanh', name='mix_inter_intra_embedding')(lig_mut_mix_inter_intra_branch)
    logger.info(f"  Dense(4, tanh): (batch, 8) @ (4, 8)^T -> (batch, 4)")
    logger.info(f"  Mix embedding output: (batch, 4)")
    
    # Combine Priority 1
    priority1_combined = Concatenate(name='priority1_combined')([final_emb, lig_mut_mix_inter_intra_branch_emb])
    logger.info(f"\n[Priority 1 Combined]")
    logger.info(f"  Concatenate[(batch, 8), (batch, 4)] along axis=-1 -> (batch, 12)")
    logger.info(f"  Math: [final_emb || mix_emb] = combined vector of length 12")
    
    # Priority 2
    logger.info("\n" + "="*80)
    logger.info("PRIORITY 2: Inter-molecular Interactions (with gating)")
    logger.info("="*80)
    logger.info(f"  Input: (batch, {feature_dims['inter_interaction']})")
    
    inter_interact_branch = Dense(48, kernel_initializer='he_normal', name='inter_dense1')(inter_interaction_input)
    logger.info(f"  Dense(48): (batch, {feature_dims['inter_interaction']}) -> (batch, 48)")
    
    inter_interact_branch = LeakyReLU(alpha=0.1, name='inter_leaky1')(inter_interact_branch)
    inter_interact_branch = BatchNormalization(name='inter_bn1')(inter_interact_branch)
    logger.info(f"  LeakyReLU + BatchNorm: -> (batch, 48)")
    
    # Gating mechanism
    inter_gate = Dense(48, activation='sigmoid', kernel_initializer='glorot_uniform', name='inter_gate')(priority1_combined)
    logger.info(f"\n[Gating Mechanism]")
    logger.info(f"  Gate: Priority1(batch, 12) -> Dense(48, sigmoid) -> (batch, 48)")
    logger.info(f"    sigmoid: f(x) = 1/(1 + e^-x), range [0, 1]")
    logger.info(f"    Purpose: Learn which features from inter_interaction to pass through")
    
    inter_gated = Multiply(name='inter_gating')([inter_interact_branch, inter_gate])
    logger.info(f"  Element-wise Multiply: (batch, 48) âŠ™ (batch, 48) -> (batch, 48)")
    logger.info(f"    Math: output[i] = inter_branch[i] * gate[i], for i=0..47")
    logger.info(f"    Effect: Gate values close to 0 suppress features, close to 1 pass them through")
    
    inter_gated = Dropout(0.1, name='inter_dropout')(inter_gated)
    inter_branch = Dense(24, kernel_initializer='he_normal', name='inter_dense2')(inter_gated)
    logger.info(f"  Dense(24): (batch, 48) -> (batch, 24)")
    
    inter_branch = LeakyReLU(alpha=0.1, name='inter_leaky2')(inter_branch)
    inter_emb = Dense(12, activation='tanh', name='inter_embedding')(inter_branch)
    logger.info(f"  Dense(12, tanh): (batch, 24) -> (batch, 12)")
    logger.info(f"  Inter embedding output: (batch, 12)")
    
    priority1_2_combined = Concatenate(name='priority1_2_combined')([priority1_combined, inter_emb])
    logger.info(f"\n[Priority 1+2 Combined]")
    logger.info(f"  Concatenate[(batch, 12), (batch, 12)] -> (batch, 24)")
    
    # Priority 3
    logger.info("\n" + "="*80)
    logger.info("PRIORITY 3: Intra-molecular Interactions (with gating)")
    logger.info("="*80)
    logger.info(f"  Input: (batch, {feature_dims['intra_interaction']})")
    
    intra_interact_branch = Dense(48, kernel_initializer='he_normal', name='intra_dense1')(intra_interaction_input)
    logger.info(f"  Dense(48): (batch, {feature_dims['intra_interaction']}) -> (batch, 48)")
    
    intra_interact_branch = LeakyReLU(alpha=0.1, name='intra_leaky1')(intra_interact_branch)
    intra_interact_branch = BatchNormalization(name='intra_bn1')(intra_interact_branch)
    logger.info(f"  LeakyReLU + BatchNorm: -> (batch, 48)")
    
    intra_gate = Dense(48, activation='sigmoid', kernel_initializer='glorot_uniform', name='intra_gate')(priority1_2_combined)
    logger.info(f"\n[Gating Mechanism]")
    logger.info(f"  Gate: Priority1+2(batch, 24) -> Dense(48, sigmoid) -> (batch, 48)")
    
    intra_gated = Multiply(name='intra_gating')([intra_interact_branch, intra_gate])
    logger.info(f"  Element-wise Multiply: (batch, 48) âŠ™ (batch, 48) -> (batch, 48)")
    
    intra_gated = Dropout(0.1, name='intra_dropout')(intra_gated)
    intra_branch = Dense(24, kernel_initializer='he_normal', name='intra_dense2')(intra_gated)
    intra_branch = LeakyReLU(alpha=0.1, name='intra_leaky2')(intra_branch)
    intra_emb = Dense(12, activation='tanh', name='intra_embedding')(intra_branch)
    logger.info(f"  Dense(12, tanh): (batch, 24) -> (batch, 12)")
    logger.info(f"  Intra embedding output: (batch, 12)")
    
    priority1_2_3_combined = Concatenate(name='priority1_2_3_combined')([priority1_2_combined, intra_emb])
    logger.info(f"\n[Priority 1+2+3 Combined]")
    logger.info(f"  Concatenate[(batch, 24), (batch, 12)] -> (batch, 36)")
    
    # Priority 4-5 (Inter features)
    logger.info("\n" + "="*80)
    logger.info("PRIORITY 4-5: Individual Inter-molecular Features (Mutation & Ligand)")
    logger.info("="*80)
    
    # Mutation inter
    logger.info(f"\n[Mutation Inter Branch]")
    logger.info(f"  Input: (batch, {feature_dims['mut_inter']})")
    mut_inter_branch = Dense(32, kernel_initializer='he_normal', name='mut_inter_dense1')(mut_inter_input)
    logger.info(f"  Dense(32): (batch, {feature_dims['mut_inter']}) -> (batch, 32)")
    
    mut_inter_branch = LeakyReLU(alpha=0.1, name='mut_inter_leaky1')(mut_inter_branch)
    mut_inter_branch = BatchNormalization(name='mut_inter_bn')(mut_inter_branch)
    
    mut_inter_gate = Dense(32, activation='sigmoid', kernel_initializer='glorot_uniform', name='mut_inter_gate')(priority1_2_3_combined)
    logger.info(f"  Gate: Priority1+2+3(batch, 36) -> Dense(32, sigmoid) -> (batch, 32)")
    
    mut_inter_gated = Multiply(name='mut_inter_gating')([mut_inter_branch, mut_inter_gate])
    logger.info(f"  Gated Multiply: (batch, 32) âŠ™ (batch, 32) -> (batch, 32)")
    
    mut_inter_gated = Dropout(0.1, name='mut_inter_dropout')(mut_inter_gated)
    mut_inter_branch = Dense(16, kernel_initializer='he_normal', name='mut_inter_dense2')(mut_inter_gated)
    mut_inter_branch = LeakyReLU(alpha=0.1, name='mut_inter_leaky2')(mut_inter_branch)
    mut_inter_emb = Dense(8, activation='tanh', name='mut_inter_embedding')(mut_inter_branch)
    logger.info(f"  Final: (batch, 32) -> Dense(16) -> Dense(8, tanh) -> (batch, 8)")
    
    # Ligand inter
    logger.info(f"\n[Ligand Inter Branch]")
    logger.info(f"  Input: (batch, {feature_dims['lig_inter']})")
    lig_inter_branch = Dense(32, kernel_initializer='he_normal', name='lig_inter_dense1')(lig_inter_input)
    logger.info(f"  Dense(32): (batch, {feature_dims['lig_inter']}) -> (batch, 32)")
    
    lig_inter_branch = LeakyReLU(alpha=0.1, name='lig_inter_leaky1')(lig_inter_branch)
    lig_inter_branch = BatchNormalization(name='lig_inter_bn')(lig_inter_branch)
    
    lig_inter_gate = Dense(32, activation='sigmoid', kernel_initializer='glorot_uniform', name='lig_inter_gate')(priority1_2_3_combined)
    logger.info(f"  Gate: Priority1+2+3(batch, 36) -> Dense(32, sigmoid) -> (batch, 32)")
    
    lig_inter_gated = Multiply(name='lig_inter_gating')([lig_inter_branch, lig_inter_gate])
    logger.info(f"  Gated Multiply: (batch, 32) âŠ™ (batch, 32) -> (batch, 32)")
    
    lig_inter_gated = Dropout(0.1, name='lig_inter_dropout')(lig_inter_gated)
    lig_inter_branch = Dense(16, kernel_initializer='he_normal', name='lig_inter_dense2')(lig_inter_gated)
    lig_inter_branch = LeakyReLU(alpha=0.1, name='lig_inter_leaky2')(lig_inter_branch)
    lig_inter_emb = Dense(8, activation='tanh', name='lig_inter_embedding')(lig_inter_branch)
    logger.info(f"  Final: (batch, 32) -> Dense(16) -> Dense(8, tanh) -> (batch, 8)")
    
    inter_combined = Concatenate(name='inter_combined')([mut_inter_emb, lig_inter_emb])
    logger.info(f"\n[Inter Combined]")
    logger.info(f"  Concatenate[(batch, 8), (batch, 8)] -> (batch, 16)")
    
    priority1_to_5_combined = Concatenate(name='priority1_to_5_combined')([priority1_2_3_combined, inter_combined])
    logger.info(f"  Priority 1-5 Combined: (batch, 36) + (batch, 16) -> (batch, 52)")
    
    # Priority 6-7 (Intra features)
    logger.info("\n" + "="*80)
    logger.info("PRIORITY 6-7: Individual Intra-molecular Features (Mutation & Ligand)")
    logger.info("="*80)
    
    # Mutation intra
    logger.info(f"\n[Mutation Intra Branch]")
    logger.info(f"  Input: (batch, {feature_dims['mut_intra']})")
    mut_intra_branch = Dense(32, kernel_initializer='he_normal', name='mut_intra_dense1')(mut_intra_input)
    logger.info(f"  Dense(32): (batch, {feature_dims['mut_intra']}) -> (batch, 32)")
    
    mut_intra_branch = LeakyReLU(alpha=0.1, name='mut_intra_leaky1')(mut_intra_branch)
    mut_intra_branch = BatchNormalization(name='mut_intra_bn')(mut_intra_branch)
    
    mut_intra_gate = Dense(32, activation='sigmoid', kernel_initializer='glorot_uniform', name='mut_intra_gate')(priority1_to_5_combined)
    logger.info(f"  Gate: Priority1-5(batch, 52) -> Dense(32, sigmoid) -> (batch, 32)")
    
    mut_intra_gated = Multiply(name='mut_intra_gating')([mut_intra_branch, mut_intra_gate])
    logger.info(f"  Gated Multiply: (batch, 32) âŠ™ (batch, 32) -> (batch, 32)")
    
    mut_intra_gated = Dropout(0.25, name='mut_intra_dropout')(mut_intra_gated)
    mut_intra_branch = Dense(16, kernel_initializer='he_normal', name='mut_intra_dense2')(mut_intra_gated)
    mut_intra_branch = LeakyReLU(alpha=0.1, name='mut_intra_leaky2')(mut_intra_branch)
    mut_intra_emb = Dense(8, activation='tanh', name='mut_intra_embedding')(mut_intra_branch)
    logger.info(f"  Final: (batch, 32) -> Dense(16) -> Dense(8, tanh) -> (batch, 8)")
    
    # Ligand intra
    logger.info(f"\n[Ligand Intra Branch]")
    logger.info(f"  Input: (batch, {feature_dims['lig_intra']})")
    lig_intra_branch = Dense(32, kernel_initializer='he_normal', name='lig_intra_dense1')(lig_intra_input)
    logger.info(f"  Dense(32): (batch, {feature_dims['lig_intra']}) -> (batch, 32)")
    
    lig_intra_branch = LeakyReLU(alpha=0.1, name='lig_intra_leaky1')(lig_intra_branch)
    lig_intra_branch = BatchNormalization(name='lig_intra_bn')(lig_intra_branch)
    
    lig_intra_gate = Dense(32, activation='sigmoid', kernel_initializer='glorot_uniform', name='lig_intra_gate')(priority1_to_5_combined)
    logger.info(f"  Gate: Priority1-5(batch, 52) -> Dense(32, sigmoid) -> (batch, 32)")
    
    lig_intra_gated = Multiply(name='lig_intra_gating')([lig_intra_branch, lig_intra_gate])
    logger.info(f"  Gated Multiply: (batch, 32) âŠ™ (batch, 32) -> (batch, 32)")
    
    lig_intra_gated = Dropout(0.25, name='lig_intra_dropout')(lig_intra_gated)
    lig_intra_branch = Dense(16, kernel_initializer='he_normal', name='lig_intra_dense2')(lig_intra_gated)
    lig_intra_branch = LeakyReLU(alpha=0.1, name='lig_intra_leaky2')(lig_intra_branch)
    lig_intra_emb = Dense(8, activation='tanh', name='lig_intra_embedding')(lig_intra_branch)
    logger.info(f"  Final: (batch, 32) -> Dense(16) -> Dense(8, tanh) -> (batch, 8)")
    
    intra_combined = Concatenate(name='intra_combined')([mut_intra_emb, lig_intra_emb])
    logger.info(f"\n[Intra Combined]")
    logger.info(f"  Concatenate[(batch, 8), (batch, 8)] -> (batch, 16)")
    
    # Final combination
    all_combined = Concatenate(name='all_combined')([
        priority1_2_3_combined,
        inter_combined,
        intra_combined
    ])
    logger.info("\n" + "="*80)
    logger.info("FINAL INTEGRATION LAYERS")
    logger.info("="*80)
    logger.info(f"  All Combined: (batch, 36) + (batch, 16) + (batch, 16) -> (batch, 68)")
    
    x = Dense(128, kernel_initializer='he_normal', name='integration_dense1')(all_combined)
    logger.info(f"  Dense(128): (batch, 68) @ (128, 68)^T -> (batch, 128)")
    
    x = LeakyReLU(alpha=0.1, name='integration_leaky1')(x)
    x = BatchNormalization(name='integration_bn1')(x)
    x = Dropout(0.3, name='integration_dropout1')(x)
    logger.info(f"  LeakyReLU + BatchNorm + Dropout(0.3): -> (batch, 128)")
    
    x = Dense(64, kernel_initializer='he_normal', name='integration_dense2')(x)
    logger.info(f"  Dense(64): (batch, 128) -> (batch, 64)")
    
    x = LeakyReLU(alpha=0.1, name='integration_leaky2')(x)
    x = BatchNormalization(name='integration_bn2')(x)
    x = Dropout(0.2, name='integration_dropout2')(x)
    logger.info(f"  LeakyReLU + BatchNorm + Dropout(0.2): -> (batch, 64)")
    
    x = Dense(32, kernel_initializer='he_normal', name='integration_dense3')(x)
    x = LeakyReLU(alpha=0.1, name='integration_leaky3')(x)
    logger.info(f"  Dense(32) + LeakyReLU: (batch, 64) -> (batch, 32)")
    
    # CRITICAL: Embedding layer for sequential model 
    embedding_layer = Dense(16, kernel_initializer='he_normal', name='embedding_layer')(x)
    embedding_output = LeakyReLU(alpha=0.1, name='embedding_output')(embedding_layer)
    logger.info(f"\n*** EMBEDDING LAYER (CRITICAL FOR RNN) ***")
    logger.info(f"  Dense(16) + LeakyReLU: (batch, 32) -> (batch, 16)")
    logger.info(f"  This 16-dim embedding will be extracted for each mutation")
    logger.info(f"  5 mutations -> 5 embeddings -> (batch, 5, 16) for RNN input")
    
    # Final prediction
    activity_head = Dense(8, kernel_initializer='he_normal', name='activity_head')(embedding_output)
    activity_head = LeakyReLU(alpha=0.1, name='activity_head_activation')(activity_head)
    activity_head = Dropout(0.2, name='activity_head_dropout')(activity_head)
    activity_output = Dense(1, activation='linear', kernel_initializer='glorot_uniform', name='activity_output')(activity_head)
    logger.info(f"\n[Activity Output Head]")
    logger.info(f"  Dense(8) + LeakyReLU + Dropout(0.2): (batch, 16) -> (batch, 8)")
    logger.info(f"  Dense(1, linear): (batch, 8) -> (batch, 1)")
    logger.info(f"  Predicts: Activity (IC50/Ki)")

    # Docking head
    docking_head = Dense(8, kernel_initializer='he_normal', name='docking_head')(embedding_output)
    docking_head = LeakyReLU(alpha=0.1, name='docking_head_activation')(docking_head)
    docking_head = Dropout(0.2, name='docking_head_dropout')(docking_head)
    docking_output = Dense(1, activation='linear', kernel_initializer='glorot_uniform', name='docking_output')(docking_head)
    logger.info(f"\n[Docking Output Head]")
    logger.info(f"  Dense(8) + LeakyReLU + Dropout(0.2): (batch, 16) -> (batch, 8)")
    logger.info(f"  Dense(1, linear): (batch, 8) -> (batch, 1)")
    logger.info(f"  Predicts: Docking Score")

    model = Model(
        inputs=[
            final_interaction_input,
            lig_mut_mix_inter_intra_input,
            inter_interaction_input,
            intra_interaction_input,
            mut_inter_input,
            lig_inter_input,
            mut_intra_input,
            lig_intra_input,
        ],
        outputs=[activity_output, docking_output],  # TWO OUTPUTS
        name='priority_hierarchical_model'
    )

    model.compile(
        optimizer=Adam(learning_rate=0.003),
        loss={
            'activity_output': 'mean_squared_error',
            'docking_output': 'mean_squared_error'
        },
        loss_weights={
            'activity_output': 1.0,      # Primary target
            'docking_output': 0.6        # Secondary target
        },
        metrics={
            'activity_output': ['mae', 'mse'],
            'docking_output': ['mae', 'mse']
        }
    )
    
    logger.info("\n" + "="*80)
    logger.info("MODEL COMPILATION COMPLETE")
    logger.info(f"  Total trainable parameters: {model.count_params():,}")
    logger.info(f"  Optimizer: Adam(lr=0.003)")
    logger.info(f"  Loss: MSE (Mean Squared Error)")
    logger.info(f"  Metrics: MAE, MSE")
    logger.info("="*80 + "\n")
    
    return model

# Recurrent neural network by looping over sequence priority weights in timesteps
# Two path apporach using Bidirectional LTSM and Bidirectional GRU
# Concatenate and continue with forward dense network

# # Forward LSTM/GRU reads the biological/mechanistic sequence:
# t=0: FULL_SMILES     -> Overall protein context
# t=1: ATP_POCKET      -> Active site structure
# t=2: P_LOOP          -> ATP binding region (glycine-rich loop)
# t=3: C_HELIX         -> Regulatory region
# t=4: DEL19           -> Exon 19 deletion region
# t=5: HINGE_LOOP      -> Hinge / gatekeeper region (T790M, C797S)
# t=6: DFG_A_LOOP      -> Activation loop (L858R)
# t=7: HRD_CAT         -> Catalytic site

# # Backward LSTM/GRU reads in reverse:
# t=7: HRD_CAT         -> Catalytic outcome
# t=6: DFG_A_LOOP      -> Activation state
# t=5: HINGE_LOOP      -> Hinge / gatekeeper state
# t=4: DEL19           -> Deletion region state
# t=3: C_HELIX         -> Regulatory input
# t=2: P_LOOP          -> Binding mechanics
# t=1: ATP_POCKET      -> Active site
# t=0: FULL_SMILES     -> Overall context

#RNN Model Receives Temporal Sequence, Each respective mutation substurcture follows seqeunce for mechanistic EFGR signnal transduction
# embedding_dim = sequential_embeddings.shape[2]  # = 16
# n_timesteps = sequential_embeddings.shape[1]    # = 8

def build_rnn_sequential_model(embedding_dim, n_timesteps=8):

    logger.info("="*80)
    logger.info("BUILDING RNN-LSTM SEQUENTIAL MODEL")
    logger.info("="*80)
    
    logger.info(f"\n--- INPUT SPECIFICATION ---")
    logger.info(f"  embedding_dim: {embedding_dim} (features per timestep)")
    logger.info(f"  n_timesteps: {n_timesteps} (number of sequential mutations)")
    logger.info(f"  Input shape: (batch_size, {n_timesteps}, {embedding_dim})")
    logger.info(f"  Interpretation: Each sample has {n_timesteps} mutations, each represented by {embedding_dim} features")
    
    sequence_input = Input(shape=(n_timesteps, embedding_dim), name='mutation_sequence')
    logger.info(f"\n[Sequence Input]")
    logger.info(f"  Shape: (batch, {n_timesteps}, {embedding_dim})")
    logger.info(f"  Example: batch=32 -> (32, {n_timesteps}, {embedding_dim})")
    
    # Bidirectional LSTM Path
    logger.info("\n" + "="*80)
    logger.info("LSTM PATH (Bidirectional)")
    logger.info("="*80)
    
    lstm_out = Bidirectional(
        LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
        name='bilstm_1'
    )(sequence_input)
    logger.info(f"\n[BiLSTM Layer 1]")
    logger.info(f"  Input: (batch, {n_timesteps}, {embedding_dim})")
    logger.info(f"  LSTM units: 128 (per direction)")
    logger.info(f"  Bidirectional: Processes sequence forward & backward")
    logger.info(f"    Forward LSTM: reads t=0,1,2,3,4,5,6,7 -> output (batch, {n_timesteps}, 128)")
    logger.info(f"    Backward LSTM: reads t=7,6,5,4,3,2,1,0 -> output (batch, {n_timesteps}, 128)")
    logger.info(f"  Concatenate: [forward || backward] -> (batch, {n_timesteps}, 256)")
    logger.info(f"  return_sequences=True: Output at each timestep")
    logger.info(f"  dropout=0.2: Drop 20% of input units")
    logger.info(f"  recurrent_dropout=0.2: Drop 20% of recurrent connections")
    logger.info(f"  Output: (batch, {n_timesteps}, 256)")
    
    lstm_out = BatchNormalization(name='bn_lstm1')(lstm_out)
    logger.info(f"  BatchNorm: Normalize across batch dimension -> (batch, {n_timesteps}, 256)")
    
    lstm_out = Bidirectional(
        LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
        name='bilstm_2'
    )(lstm_out)
    logger.info(f"\n[BiLSTM Layer 2]")
    logger.info(f"  Input: (batch, {n_timesteps}, 256)")
    logger.info(f"  LSTM units: 64 (per direction)")
    logger.info(f"  return_sequences=False: Only output at final timestep")
    logger.info(f"    Forward: final hidden state -> (batch, 64)")
    logger.info(f"    Backward: final hidden state -> (batch, 64)")
    logger.info(f"  Concatenate: [forward || backward] -> (batch, 128)")
    logger.info(f"  Output: (batch, 128) [sequence collapsed to single vector]")
    
    lstm_out = BatchNormalization(name='bn_lstm2')(lstm_out)
    logger.info(f"  BatchNorm: -> (batch, 128)")
    
    # Bidirectional GRU Path
    logger.info("\n" + "="*80)
    logger.info("GRU PATH (Bidirectional, Parallel to LSTM)")
    logger.info("="*80)
    
    gru_out = Bidirectional(
        GRU(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2),
        name='bigru_1'
    )(sequence_input)
    logger.info(f"\n[BiGRU Layer 1]")
    logger.info(f"  Input: (batch, {n_timesteps}, {embedding_dim}) [same as LSTM input]")
    logger.info(f"  GRU units: 128 (per direction)")
    logger.info(f"  GRU vs LSTM: Simpler architecture, fewer gates (update & reset vs input, forget, output)")
    logger.info(f"    Forward GRU: reads t=0,1,2,3,4,5,6,7 -> (batch, {n_timesteps}, 128)")
    logger.info(f"    Backward GRU: reads t=7,6,5,4,3,2,1,0 -> (batch, {n_timesteps}, 128)")
    logger.info(f"  Concatenate: -> (batch, {n_timesteps}, 256)")
    logger.info(f"    where z_t is update gate, hÌƒ_t is candidate activation")
    logger.info(f"  Output: (batch, {n_timesteps}, 256)")
    
    gru_out = BatchNormalization(name='bn_gru1')(gru_out)
    logger.info(f"  BatchNorm: -> (batch, {n_timesteps}, 256)")
    
    gru_out = Bidirectional(
        GRU(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2),
        name='bigru_2'
    )(gru_out)
    logger.info(f"\n[BiGRU Layer 2]")
    logger.info(f"  Input: (batch, {n_timesteps}, 256)")
    logger.info(f"  GRU units: 64 (per direction)")
    logger.info(f"  return_sequences=False: Only final timestep output")
    logger.info(f"  Concatenate: [forward || backward] -> (batch, 128)")
    logger.info(f"  Output: (batch, 128) [sequence collapsed]")
    
    gru_out = BatchNormalization(name='bn_gru2')(gru_out)
    logger.info(f"  BatchNorm: -> (batch, 128)")
    
    # Combine LSTM and GRU
    combined = Concatenate(name='lstm_gru_combined')([lstm_out, gru_out])
    logger.info("\n" + "="*80)
    logger.info("COMBINING LSTM AND GRU PATHS")
    logger.info("="*80)
    logger.info(f"  LSTM output: (batch, 128)")
    logger.info(f"  GRU output: (batch, 128)")
    logger.info(f"  Concatenate: [(batch, 128) || (batch, 128)] -> (batch, 256)")
    logger.info(f"  Rationale: LSTM captures long-term dependencies, GRU provides complementary patterns")
    logger.info(f"  Combined representation: (batch, 256)")
    
    # Dense layers
    logger.info("\n" + "="*80)
    logger.info("DENSE INTEGRATION LAYERS")
    logger.info("="*80)
    
    x = Dense(128, activation='relu', name='rnn_dense1')(combined)
    logger.info(f"\n[Dense Layer 1]")
    logger.info(f"  Dense(128, relu): (batch, 256) @ (128, 256)^T -> (batch, 128)")
    logger.info(f"  ReLU: f(x) = max(0, x) - introduces non-linearity")
    
    x = BatchNormalization(name='rnn_bn1')(x)
    x = Dropout(0.3, name='rnn_dropout1')(x)
    logger.info(f"  BatchNorm + Dropout(0.3): -> (batch, 128)")
    logger.info(f"    Dropout rate increased to 0.3 to prevent overfitting on sequential patterns")
    
    x = Dense(64, activation='relu', name='rnn_dense2')(x)
    logger.info(f"\n[Dense Layer 2]")
    logger.info(f"  Dense(64, relu): (batch, 128) -> (batch, 64)")
    
    x = BatchNormalization(name='rnn_bn2')(x)
    x = Dropout(0.2, name='rnn_dropout2')(x)
    logger.info(f"  BatchNorm + Dropout(0.2): -> (batch, 64)")
    
    x = Dense(32, activation='relu', name='rnn_dense3')(x)
    logger.info(f"\n[Dense Layer 3]")
    logger.info(f"  Dense(32, relu): (batch, 64) -> (batch, 32)")
    
    x = Dropout(0.1, name='rnn_dropout3')(x)
    logger.info(f"  Dropout(0.1): -> (batch, 32)")
    
    # Final output
    activity_final = Dense(16, activation='relu', name='activity_final_head')(x)
    activity_final = Dropout(0.15, name='activity_final_dropout')(activity_final)
    activity_output = Dense(1, activation='linear', name='final_activity_output')(activity_final)

    # Docking output
    docking_final = Dense(16, activation='relu', name='docking_final_head')(x)
    docking_final = Dropout(0.15, name='docking_final_dropout')(docking_final)
    docking_output = Dense(1, activation='linear', name='final_docking_output')(docking_final)

    logger.info("\n" + "="*80)
    logger.info("FINAL OUTPUT - MULTI-TASK")
    logger.info("="*80)
    logger.info(f"  Activity Head: Dense(16, relu) + Dropout(0.15) -> Dense(1, linear)")
    logger.info(f"  Docking Head: Dense(16, relu) + Dropout(0.15) -> Dense(1, linear)")
    logger.info(f"  Output: Two predictions [activity, docking]")

    model = Model(inputs=sequence_input, outputs=[activity_output, docking_output], name='rnn_sequential_model')

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss={
            'final_activity_output': 'mean_squared_error',
            'final_docking_output': 'mean_squared_error'
        },
        loss_weights={
            'final_activity_output': 1.0,
            'final_docking_output': 0.7
        },
        metrics={
            'final_activity_output': ['mae', 'mse'],
            'final_docking_output': ['mae', 'mse']
        }
    )
    
    logger.info("\n" + "="*80)
    logger.info("RNN MODEL COMPILATION COMPLETE")
    logger.info("="*80)
    logger.info(f"  Input: (batch, {n_timesteps}, {embedding_dim})")
    logger.info(f"  Output: (batch, 1)")
    logger.info(f"  Total trainable parameters: {model.count_params():,}")
    logger.info(f"  Optimizer: Adam(lr=0.001)")
    logger.info(f"  Loss: MSE (Mean Squared Error)")
    logger.info(f"  Metrics: MAE, MSE")
    logger.info("\n--- WORKFLOW SUMMARY ---")
    logger.info(f"  1. Each mutation site -> Hierarchical Model -> 16-dim embedding")
    logger.info(f"  2. Stack 5 embeddings -> (batch, 5, 16) sequence")
    logger.info(f"  3. BiLSTM + BiGRU process temporal dependencies")
    logger.info(f"  4. Combine representations -> Dense layers")
    logger.info(f"  5. Output: Single binding affinity prediction")
    logger.info("="*80 + "\n")
    
    print("\n" + "="*80)
    print("RNN-LSTM SEQUENTIAL MODEL BUILT SUCCESSFULLY")
    print(f"  Input shape: (batch, {n_timesteps} timesteps, {embedding_dim} features)")
    print(f"  Total parameters: {model.count_params():,}")
    print("="*80)
    
    return model



# ============================================================================
# MAIN SCRIPT
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
        logger.debug(f" Feature dict for {site_name}: feature type: {type(feature_dict)}")
        logger.debug(f" Feature dict for {site_name}: feature dict: {feature_dict}")
        
    
    # Find common valid indices
    common_valid_indices = set.intersection(*all_valid_indices)
    common_valid_indices = sorted(list(common_valid_indices))
    logger.info(f"Common valid indices across all sites: {common_valid_indices}")
    
    print(f"\n{'='*80}")
    print(f"Common valid samples across all sites: {len(common_valid_indices)}")
    print(f"{'='*80}")
    
    if len(common_valid_indices) == 0:
        print("\n❌ ERROR: No samples remain after filtering!")
        sys.exit(1)
    
    # Filter to common indices
    for i, feature_dict in enumerate(all_feature_dicts):
        site_valid_idx = feature_dict['valid_indices']
        mask = np.isin(site_valid_idx, common_valid_indices)
        
        for key in ['lig_inter', 'mut_inter', 'inter_interaction', 
                    'lig_intra', 'mut_intra', 'intra_interaction', 
                    'lig_mut_mix_inter_intra', 'final_fp_interaction']:
            all_feature_dicts[i][key] = feature_dict[key][mask]

    # use log1p (log(1+x)) to avoid errors with 0 values
    # Get y_train
    # Get y_train for activity (IC50/Ki values)
    y_train1 = activity_values_valid.iloc[common_valid_indices].values
    y_train1 = np.log1p(y_train1)  # use log1p (log(1+x)) to avoid errors with 0 values

    y_scaler1 = StandardScaler()
    y_train_scaled1 = y_scaler1.fit_transform(y_train1.reshape(-1, 1)).flatten()

    # Get y_train2 for docking scores
    y_train2 = activity_values2_valid[common_valid_indices]


    y_scaler2 = StandardScaler()
    y_train_scaled2 = y_scaler2.fit_transform(y_train2.reshape(-1, 1)).flatten()

    print(f"\nTraining samples: {len(y_train1)}")
    print(f"y_train1 (activity) - min: {y_train1.min():.2f}, max: {y_train1.max():.2f}, mean: {y_train1.mean():.2f}")
    print(f"y_train2 (docking) - min: {y_train2.min():.2f}, max: {y_train2.max():.2f}, mean: {y_train2.mean():.2f}")
    
    # ===== STAGE 2: TRAIN HIERARCHICAL MODELS & EXTRACT EMBEDDINGS =====
    print("\n" + "="*80)
    print("STAGE 2: TRAIN HIERARCHICAL MODELS & EXTRACT EMBEDDINGS")
    print("="*80)
    
    all_scalers = []
    all_embeddings = []
    
    for site_idx, (site_name, _) in enumerate(mutation_sites):
        print(f"\n{'='*80}")
        print(f"Site {site_idx+1}/8: {site_name}")
        print(f"{'='*80}")
        
        feature_dict = all_feature_dicts[site_idx]
        
        # Normalize
        scalers = {}
        scaled_features = {}
        
        for key in ['lig_inter', 'mut_inter', 'inter_interaction', 
                    'lig_intra', 'mut_intra', 'intra_interaction', 
                    'lig_mut_mix_inter_intra', 'final_fp_interaction']:
            scalers[key] = StandardScaler()
            scaled_features[key] = scalers[key].fit_transform(feature_dict[key])
            print(f"  {key:25s}: mean={scaled_features[key].mean():.4f}, std={scaled_features[key].std():.4f}")
        
        all_scalers.append(scalers)
        logger.info(f"all scalers list: {all_scalers}")
        
        # Build model
        feature_dims = {
            'lig_inter': scaled_features['lig_inter'].shape[1],
            'mut_inter': scaled_features['mut_inter'].shape[1],
            'inter_interaction': scaled_features['inter_interaction'].shape[1],
            'lig_intra': scaled_features['lig_intra'].shape[1],
            'mut_intra': scaled_features['mut_intra'].shape[1],
            'intra_interaction': scaled_features['intra_interaction'].shape[1],
            'lig_mut_mix_inter_intra': scaled_features['lig_mut_mix_inter_intra'].shape[1],
            'final_fp_interaction': scaled_features['final_fp_interaction'].shape[1],
        }
        
        model = build_priority_hierarchical_model(feature_dims)

        if site_idx == 0:  # Only print for first site to avoid clutter
            print(f"\n{'='*80}")
            print(f"HIERARCHICAL MODEL ARCHITECTURE ({site_name})")
            print(f"{'='*80}")
            model.summary()
            print(f"{'='*80}\n")
            
        checkpoint = ModelCheckpoint(
            f'hierarchical_model_{site_name}.h5',
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
        
        print(f"\nTraining {site_name} model...")
        history = model.fit(
            x=[
                scaled_features['final_fp_interaction'],
                scaled_features['lig_mut_mix_inter_intra'],
                scaled_features['inter_interaction'],
                scaled_features['intra_interaction'],
                scaled_features['mut_inter'],
                scaled_features['lig_inter'],
                scaled_features['mut_intra'],
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
        
        print(f"\n✓ {site_name} training complete! Best val_loss: {min(history.history['val_loss']):.4f}")
        
        # CRITICAL FIX: Extract embeddings from 'embedding_output' layer
        # This layer comes BEFORE the final activity prediction
        # It contains the learned representation that predicts activity
        embedding_model = Model(
            inputs=model.inputs,
            outputs=model.get_layer('embedding_output').output,
            name=f'embedding_model_{site_name}'
        )
        
        print(f"\n✓ Extracting embeddings from 'embedding_output' layer...") #Extract NOT predict embeddings
        embeddings = embedding_model.predict([
            scaled_features['final_fp_interaction'],
            scaled_features['lig_mut_mix_inter_intra'],
            scaled_features['inter_interaction'],
            scaled_features['intra_interaction'],
            scaled_features['mut_inter'],
            scaled_features['lig_inter'],
            scaled_features['mut_intra'],
            scaled_features['lig_intra'],
        ], verbose=0)
        
        all_embeddings.append(embeddings)
        logger.info(f"all embeddings list: {all_embeddings}")
        print(f"  Embeddings shape: {embeddings.shape} (n_samples, embedding_dim)")
    
    # ===== STAGE 3: TRAIN RNN-LSTM MODEL =====
    print("\n" + "="*80)
    print("STAGE 3: TRAIN RNN-LSTM SEQUENTIAL MODEL")
    print("="*80)
    
    # Stack embeddings: (n_samples, 8 timesteps, 16 features)
    sequential_embeddings = np.stack(all_embeddings, axis=1)
    print(f"Sequential embeddings shape: {sequential_embeddings.shape}")
    
    embedding_dim = sequential_embeddings.shape[2]
    n_timesteps = sequential_embeddings.shape[1]
    
    rnn_model = build_rnn_sequential_model(embedding_dim, n_timesteps)

    print(f"\n{'='*80}")
    print("RNN-LSTM MODEL ARCHITECTURE")
    print(f"{'='*80}")
    rnn_model.summary()
    print(f"{'='*80}\n")
    
    rnn_checkpoint = ModelCheckpoint(
        'rnn_sequential_model.h5',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    )
    
    rnn_early_stop = EarlyStopping(
        monitor='val_loss',
        patience=40,
        restore_best_weights=True,
        verbose=1
    )
    
    print("\nTraining RNN-LSTM model...")
    rnn_history = rnn_model.fit(
        x=sequential_embeddings,
        y={
            'final_activity_output': y_train_scaled1,
            'final_docking_output': y_train_scaled2
        },
        epochs=150,
        batch_size=32,
        validation_split=0.2,
        callbacks=[rnn_early_stop, rnn_checkpoint],
        verbose=1
    )
    
    print(f"\n✓ RNN-LSTM training complete! Best val_loss: {min(rnn_history.history['val_loss']):.4f}")

    import pickle

    # ===== SAVE SCALERS FOR FUTURE PREDICTIONS =====
    print("\n" + "="*80)
    print("SAVING SCALERS")
    print("="*80)

    # Save feature scalers (one set per mutation site)
    with open('feature_scalers.pkl', 'wb') as f:
        pickle.dump(all_scalers, f)
    print("✓ Feature scalers saved to 'feature_scalers.pkl'")

    # Save y scalers
    with open('y_scalers.pkl', 'wb') as f:
        pickle.dump({'y_scaler1': y_scaler1, 'y_scaler2': y_scaler2}, f)
    print("✓ Y scalers saved to 'y_scalers.pkl'")

    # Save mutation profiles (optional but useful)
    unique_mutation_profiles.to_csv('mutation_profiles.csv', index=False)
    print("✓ Mutation profiles saved to 'mutation_profiles.csv'")

    print("="*80)
    
    # Plot training history
    print("\n" + "="*80)
    print("PLOTTING TRAINING HISTORY")
    print("="*80)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(rnn_history.history['loss'], label='Train Loss', linewidth=2, color='#2E86AB')
    axes[0].plot(rnn_history.history['val_loss'], label='Val Loss', linewidth=2, color='#A23B72')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss (MSE)', fontsize=12)
    axes[0].set_title('RNN-LSTM Sequential Model - Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(rnn_history.history['final_activity_output_mae'], label='Train Activity MAE', linewidth=2, color='#2E86AB')
    axes[1].plot(rnn_history.history['val_final_activity_output_mae'], label='Val Activity MAE', linewidth=2, color='#A23B72')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('MAE', fontsize=12)
    axes[1].set_title('RNN-LSTM Sequential Model - MAE', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('rnn_training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("✓ Training history plot saved: rnn_training_history.png")
    
    print("\n" + "="*80)
    print("EXECUTION COMPLETE")
    print("="*80)
    print("✓ 8 Hierarchical models: hierarchical_model_*.h5")
    print("✓ RNN model: rnn_sequential_model.h5")
    print("✓ Control predictions: control_predictions_rnn.csv")
    print("✓ Drug predictions: drug_predictions_rnn.csv")
    print("✓ Training plot: rnn_training_history.png")
    print("="*80)


if __name__ == "__main__":
    print("\n" + "="*80)
    print("STARTING RNN-LSTM INTEGRATED EXECUTION v6")
    print("="*80)
    try:
        main()
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

#TODO
# 1. Increase precision on custom features
# 2. Expand on similarity and fingerprint features
# 3. Increased EGFR TKI generations dataset 
# 4. Train on y_train with docking scores (kiv to train on 2D/3D methods)
# 5. Expand the control drug dataset with ALK, MET, Her2, BRAF, KRAS, Multikinase known inhibitors to test specificity 
# 6. Expand on drug dataset to test for new potential ligands 