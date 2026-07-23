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
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import tensorflow as tf
from tensorflow.keras.models import load_model

# PyTorch and Transformers
import torch
from torch.utils.data import DataLoader, TensorDataset
from transformers import AutoTokenizer, AutoModel

# Logging
from loguru import logger
logger.remove()
logger.add(sys.stderr, level="INFO")

print("=" * 80)
print("PREDICTION SCRIPT FOR SIMPLE CHEMBERTA BENCHMARK MODEL")
print("=" * 80)

# ============================================================================
# CHEMBERTA AND ENCODING UTILITIES
# ============================================================================
def get_device():
    try:
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    except:
        return torch.device('cpu')

def load_chemberta(model_name='seyonec/ChemBERTa-zinc-base-v1', device=None):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    if device is None:
        device = get_device()
    model.to(device)
    model.eval()
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

# ============================================================================
# EVALUATION AND PLOTTING FUNCTION
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
    print(f"Loading prediction data from: {input_csv}")
    df_pred = pd.read_csv(input_csv)
    df_pred.columns = df_pred.columns.str.strip()
    
    if not all(col in df_pred.columns for col in ['smiles', 'tkd']):
        raise ValueError(f"Input CSV must contain 'smiles' and 'tkd' columns. Found: {df_pred.columns.tolist()}")
    
    # Dataset fix: normalize whitespace in 'tkd' (e.g. a stray leading-space variant of
    # 'l858r/t790m/c797s triple' would otherwise be treated as an unseen mutation class).
    df_pred['tkd'] = df_pred['tkd'].astype(str).str.strip()
    
    has_ground_truth = 'standard value' in df_pred.columns and 'dock' in df_pred.columns

    print("\nLoading simple benchmark model...")
    try:
        model = load_model(os.path.join(model_dir, 'chemberta_simple_model.h5'), compile=False)
    except FileNotFoundError:
        print("Error: chemberta_simple_model.h5 not found in model directory.")
        return None
        
    print("Loading scalers and encoders...")
    try:
        with open(os.path.join(model_dir, 'benchmark_simple_scalers.pkl'), 'rb') as f:
            scalers = pickle.load(f)
            y_scaler_act = scalers['y_scaler_act']
            y_scaler_dock = scalers['y_scaler_dock']
            chem_emb_scaler = scalers['chem_emb_scaler']
            
        with open(os.path.join(model_dir, 'benchmark_simple_le.pkl'), 'rb') as f:
            le = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Error loading scalers/encoders: {e}")
        return None

    tokenizer, chem_model, device = load_chemberta()

    # Drop missing smiles/tkd
    df_pred = df_pred.dropna(subset=['smiles', 'tkd']).reset_index(drop=True)
    N_pred = len(df_pred)
    print(f"Valid prediction samples: {N_pred}")

    # Embed smiles
    print("\nComputing ChemBERTa embeddings for prediction set...")
    smi_list = df_pred['smiles'].astype(str).tolist()
    embs_raw = get_chemberta_embeddings(smi_list, tokenizer, chem_model, device)
    lig_embs = chem_emb_scaler.transform(embs_raw)

    # Encode mutations
    print("Encoding mutation categories...")
    mut_onehot = encode_mutation_onehot(df_pred['tkd'], le)

    print("Making predictions...")
    preds = model.predict([lig_embs, mut_onehot], verbose=0)
    
    pred_activity = np.expm1(y_scaler_act.inverse_transform(preds[0].reshape(-1, 1)).flatten())
    pred_docking = y_scaler_dock.inverse_transform(preds[1].reshape(-1, 1)).flatten()

    all_results = []
    for idx, row in df_pred.iterrows():
        res = {
            'smiles': row['smiles'],
            'tkd': row['tkd'],
            'predicted_activity': pred_activity[idx],
            'predicted_docking': pred_docking[idx]
        }
        if has_ground_truth:
            res['actual_activity'] = row['standard value']
            res['actual_docking'] = row['dock']
            
        # Keep other columns
        for col in df_pred.columns:
            if col not in res and col not in ['smiles', 'tkd', 'standard value', 'dock']:
                res[col] = row[col]
        all_results.append(res)
        
    df_results = pd.DataFrame(all_results)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    output_path = os.path.join(output_dir, 'predictions_benchmark_simple.csv')
    df_results.to_csv(output_path, index=False)
    print(f"\n✓ Predictions saved to: {output_path}")
    print(f"✓ Total predictions: {len(df_results)}")

    if has_ground_truth and len(df_results) > 0:
        evaluate_and_plot(df_results, output_dir, 'benchmark_simple')

    return df_results

# ============================================================================
# MAIN EXECUTION
# ============================================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Predictions for Simple ChemBERTa Benchmark model')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    parser.add_argument('--model_dir', type=str, default='.', help='Model directory')
    parser.add_argument('--output_dir', type=str, default='.', help='Output directory')
    
    args = parser.parse_args()
    results = make_predictions(args.input, args.model_dir, args.output_dir)
    print(f"\n✓ Complete! Total predictions: {len(results)}")
