#!/usr/bin/env python3


import os
import sys
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr

# RDKit imports
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski, MolSurf, GraphDescriptors, Fragments
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit import DataStructs, RDLogger

# TensorFlow/Keras imports
from tensorflow.keras.models import load_model

RDLogger.DisableLog('rdApp.*')
np.random.seed(42)

print("="*80)
print("PREDICTION SCRIPT FOR DUMMY_PHYSCHEM_5F2")
print("Aligned with Training Logic")
print("="*80)

# ============================================================================
# FEATURE GENERATION FUNCTIONS (Copied exactly from dummy_physchem_5f2.py)
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
    Make predictions using the trained dummy_physchem_5f2 model.
    
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
    
    # Check required columns
    required_cols = ['smiles', 'tkd']
    if not all(col in df_pred.columns for col in required_cols):
        raise ValueError(f"Input CSV must contain 'smiles' and 'tkd' columns. Found: {df_pred.columns.tolist()}")
    
    # Check if ground truth exists
    has_ground_truth = 'standard value' in df_pred.columns and 'dock' in df_pred.columns
    
    # Load Model Files
    print("\nLoading model files...")
    try:
        model = load_model(os.path.join(model_dir, 'feedforward_model.h5'))
        
        with open(os.path.join(model_dir, 'mutant_encoder.pkl'), 'rb') as f:
            mutant_encoder = pickle.load(f)
            
        with open(os.path.join(model_dir, 'mutant_mapping.pkl'), 'rb') as f:
            mutant_mapping = pickle.load(f)
            
        with open(os.path.join(model_dir, 'feature_scalers.pkl'), 'rb') as f:
            feature_scalers = pickle.load(f)
        
        with open(os.path.join(model_dir, 'y_scalers.pkl'), 'rb') as f:
            y_scalers = pickle.load(f)
            
        y_scaler1 = y_scalers['y_scaler1']
        y_scaler2 = y_scalers['y_scaler2']
        
    except FileNotFoundError as e:
        print(f"Error loading model files: {e}")
        print("Please ensure 'feedforward_model.h5', 'mutant_encoder.pkl', 'mutant_mapping.pkl', 'feature_scalers.pkl', and 'y_scalers.pkl' are in the model directory.")
        return None

    print(f"Total prediction samples: {len(df_pred)}")
    
    # Prepare batch data
    lig_inter_list = []
    lig_intra_list = []
    mutant_id_list = []
    valid_indices = []
    
    print("Generating features...")
    for idx, row in df_pred.iterrows():
        lig_smiles = row['smiles']
        mutant_name = row['tkd']
        
        # Validate SMILES
        if pd.isna(lig_smiles) or lig_smiles == '':
            print(f"Warning: Empty SMILES at row {idx}, skipping")
            continue

        # Validate Mutation
        if pd.isna(mutant_name):
            print(f"Warning: Empty mutation at row {idx}, skipping")
            continue
            
        # 1. Generate Lib Inputs
        lig_inter = generate_lig_inter_features(lig_smiles)
        lig_intra = generate_lig_intra_features(lig_smiles)
        
        if lig_inter is None or lig_intra is None:
            print(f"Warning: Could not generate features for row {idx} (SMILES: {lig_smiles}), skipping")
            continue
            
        # 2. Encode Mutation
        # Training script uses simple integer encoding relative to sorted unique values in training
        if mutant_name in mutant_mapping:
            mut_id = mutant_mapping[mutant_name]
        else:
            print(f"Warning: Mutation '{mutant_name}' not seen in training. Using default ID 0.")
            mut_id = 0 # Default fallback
            
        lig_inter_list.append(lig_inter)
        lig_intra_list.append(lig_intra)
        mutant_id_list.append(mut_id)
        valid_indices.append(idx)
        
        if (idx + 1) % 100 == 0:
            print(f"Processed {idx + 1}/{len(df_pred)} samples")
            
    if not valid_indices:
        print("Error: No valid samples found to predict.")
        return None
        
    # Convert to numpy arrays
    X_lig_inter = np.array(lig_inter_list)
    X_lig_intra = np.array(lig_intra_list)
    X_mutant = np.array(mutant_id_list)
    
    # Scale features
    # Note: Training script scales lig_inter and lig_intra. Mutant ID is used as is (Embedding layer handles it).
    X_lig_inter_scaled = feature_scalers['lig_inter'].transform(X_lig_inter)
    X_lig_intra_scaled = feature_scalers['lig_intra'].transform(X_lig_intra)
    
    print(f"Predicting for {len(valid_indices)} samples...")
    
    # Make Prediction
    # Model inputs: [mutant_input, inter_input, intra_input]
    predictions = model.predict(
        [X_mutant, X_lig_inter_scaled, X_lig_intra_scaled],
        verbose=1
    )
    
    # Raw outputs
    pred_activity_scaled = predictions[0].flatten()
    pred_docking_scaled = predictions[1].flatten()
    
    # Inverse Transform
    # Activity: log1p -> scaler -> model -> scaler inverse -> expm1
    pred_activity_log1p = y_scaler1.inverse_transform(pred_activity_scaled.reshape(-1, 1)).flatten()
    pred_activity = np.expm1(pred_activity_log1p)
    
    # Docking: scaler -> model -> scaler inverse
    pred_docking = y_scaler2.inverse_transform(pred_docking_scaled.reshape(-1, 1)).flatten()
    
    # Store Results
    results = []
    for i, original_idx in enumerate(valid_indices):
        row = df_pred.iloc[original_idx]
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
                 
        results.append(res)
        
    df_results = pd.DataFrame(results)
    
    # Save output
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, 'predictions_dummy_physchem_5f2.csv')
    df_results.to_csv(output_path, index=False)
    print(f"\n✓ Predictions saved to: {output_path}")
    
    # Evaluation (if ground truth exists)
    if has_ground_truth and len(df_results) > 0:
        evaluate_and_plot(df_results, output_dir, 'dummy_physchem_5f2')
        
    return df_results

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Make predictions using trained dummy_physchem_5f2 model')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file with SMILES and mutation data')
    parser.add_argument('--model_dir', type=str, default='.', help='Directory containing model files')
    parser.add_argument('--output_dir', type=str, default='.', help='Directory to save outputs')
    
    args = parser.parse_args()
    
    results = make_predictions(args.input, args.model_dir, args.output_dir)
    
    if results is not None:
        print(f"\n✓ Complete! Total predictions: {len(results)}")
        print(f"✓ Mutations covered: {results['tkd'].nunique()}")
