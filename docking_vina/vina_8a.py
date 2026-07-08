import os
import csv
from vina import Vina
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# Receptor configurations - PDB structures
receptors_pdb = {
    '7TVD.pdbqt': {  # Exon 19 deletion
        'center': [-60.354, 7.541, 25.633],
        'box_size': [52, 74, 64],
        'mutant': 'del19'
    },
    '6LUD.pdbqt': {  # l858r/t790m/c797s triple mutant
        'center': [-57.618, -7.747, -24.514],
        'box_size': [48, 76, 58],
        'mutant': 'l858r_t790m_c797s'
    },
    '9FQP.pdbqt': {  # exon 20 insertions
        'center': [1.609, 14.072, -19.940],
        'box_size': [50, 62, 68],
        'mutant': '20ins'
    },
    '5XDL.pdbqt': {  # l858r mutant
        'center': [-57.598, -7.619, -23.749],
        'box_size': [50, 70, 66],
        'mutant': 'l858r'
    },
    '3W2P.pdbqt': {  # exon l858r_t790m double mutant
        'center': [136.872, 25.399, 57.859],
        'box_size': [70, 68, 52],
        'mutant': 'l858r_t790m'
    },
    '9FRD.pdbqt': {  # wild egfr
        'center': [-58.803, 24.197, -9.581],
        'box_size': [48, 66, 78],
        'mutant': 'wild'
    },
}

# AlphaFold structures
receptors_fold = {
    'fold_del19_T790M.pdbqt': {
        'center': [-1.051, -1.109, 2.010],
        'box_size': [60, 50, 72],
        'mutant': 'del19_t790m'
    },
    'fold_del19_T790M_C797S.pdbqt': {
        'center': [0.091, 0.402, 0.511],
        'box_size': [64, 48, 56],
        'mutant': 'del19_t790m_c797s'
    },
    'fold_19del.pdbqt': {
        'center': [-0.129, 0.259, 0.133],
        'box_size': [60, 54, 56],
        'mutant': 'del19'
    },
    'fold_l858r_T790M.pdbqt': {
        'center': [0.032, -0.118, -2.257],
        'box_size': [62, 50, 54],
        'mutant': 'l858r_t790m'
    },
    'fold_l858r_T790M_C797S.pdbqt': {
        'center': [-1.223, 0.540, -0.467],
        'box_size': [54, 58, 64],
        'mutant': 'l858r_t790m_c797s'
    },
    'fold_l858r.pdbqt': {
        'center': [0.666, 1.305, 0.845],
        'box_size': [66, 56, 60],
        'mutant': 'l858r'
    },
    'fold_20ins.pdbqt': {
        'center': [0.316, -0.438, 0.733],
        'box_size': [62, 56, 56],
        'mutant': '20ins'
    },
    'fold_wild.pdbqt': {
        'center': [-1.210, -1.807, -1.739],
        'box_size': [74, 60, 46],
        'mutant': 'wild'
    },
}

# Create folders
os.makedirs("temp_ligands", exist_ok=True)
os.makedirs("docked_poses", exist_ok=True)

# Read ligands from file 
ligands = []
with open('ligand7b.txt') as f:
    lines = [line.strip() for line in f if line.strip()]
    if lines and lines[0].lower().startswith(('id,', 'ligand', 'name')):
        lines = lines[1:]
    
    for line in lines:
        if ',' in line:
            parts = line.split(',', 1)
            ligand_id = parts[0].strip()
            smiles = parts[1].strip()
            ligands.append((ligand_id, smiles))

# Prepare CSV file
csv_file = 'docking_results.csv'
with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Ligand_ID', 'Mutant', 'Receptor', 'Source', 'Docking_Score', 
                     'Optimized_Score', 'Avg_Docking_Score', 'Avg_Optimized_Score'])

# Process each ligand
for i, (ligand_id, smiles) in enumerate(ligands):
    print(f"\n=== Processing {ligand_id}: {smiles} ===")

    # Display 2D structure
    mol_rdkit = Chem.MolFromSmiles(smiles)
    if mol_rdkit:
        img = Draw.MolToImage(mol_rdkit, size=(300, 300))
        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"{ligand_id}")
        plt.savefig(f"temp_ligands/{ligand_id}_2D.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  2D structure saved")

    mol_name = f"ligand_{i}"
    mol_mol_path = f"temp_ligands/{mol_name}.mol"
    mol_pdbqt = f"temp_ligands/{mol_name}.pdbqt"

    # Convert SMILES to MOL (3D)
    os.system(f'obabel -:"{smiles}" -O {mol_mol_path} --gen3d')

    # Visualize 3D structure
    rdkit_mol_3d = Chem.MolFromMolFile(mol_mol_path, removeHs=False)
    if rdkit_mol_3d:
        img = Draw.MolToImage(rdkit_mol_3d, size=(300, 300))
        plt.figure(figsize=(5, 5))
        plt.imshow(img)
        plt.axis("off")
        plt.title(f"3D {ligand_id}")
        plt.savefig(f"temp_ligands/{ligand_id}_3D.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  3D structure saved")

    # Convert MOL to PDBQT
    os.system(f'obabel {mol_mol_path} -O {mol_pdbqt} --partialcharge gasteiger')

    # Store results for averaging
    results_by_mutant = {}

    # Dock against PDB receptors
    for receptor_file, config in receptors_pdb.items():
        mutant = config['mutant']
        print(f"\n  Docking {ligand_id} to {receptor_file} ({mutant})...")
        
        v = Vina(sf_name='vina')
        v.set_receptor(rigid_pdbqt_filename=receptor_file)
        v.set_ligand_from_file(mol_pdbqt)
        v.compute_vina_maps(center=config['center'], box_size=config['box_size'])
        v.dock(exhaustiveness=8)
        
        score = v.score()[0]
        optimized_score = v.optimize()[0]
        print(f"  Score: {score:.3f} kcal/mol, Optimized: {optimized_score:.3f} kcal/mol")
        
        # Save pose
        out_file = f"docked_poses/{ligand_id}_{receptor_file.replace('.pdbqt', '')}.pdbqt"
        v.write_poses(out_file, n_poses=1, overwrite=True)
        
        # Store results
        if mutant not in results_by_mutant:
            results_by_mutant[mutant] = {'pdb': {}, 'fold': {}}
        results_by_mutant[mutant]['pdb'] = {
            'receptor': receptor_file,
            'dock_score': score,
            'opt_score': optimized_score
        }

    # Dock against AlphaFold receptors
    for receptor_file, config in receptors_fold.items():
        mutant = config['mutant']
        print(f"\n  Docking {ligand_id} to {receptor_file} ({mutant})...")
        
        v = Vina(sf_name='vina')
        v.set_receptor(rigid_pdbqt_filename=receptor_file)
        v.set_ligand_from_file(mol_pdbqt)
        v.compute_vina_maps(center=config['center'], box_size=config['box_size'])
        v.dock(exhaustiveness=8)
        
        score = v.score()[0]
        optimized_score = v.optimize()[0]
        print(f"  Score: {score:.3f} kcal/mol, Optimized: {optimized_score:.3f} kcal/mol")
        
        # Save pose
        out_file = f"docked_poses/{ligand_id}_{receptor_file.replace('.pdbqt', '')}.pdbqt"
        v.write_poses(out_file, n_poses=1, overwrite=True)
        
        # Store results
        if mutant not in results_by_mutant:
            results_by_mutant[mutant] = {'pdb': {}, 'fold': {}}
        results_by_mutant[mutant]['fold'] = {
            'receptor': receptor_file,
            'dock_score': score,
            'opt_score': optimized_score
        }

    # Write results to CSV with averages
    with open(csv_file, 'a', newline='') as f:
        writer = csv.writer(f)
        
        for mutant, data in sorted(results_by_mutant.items()):
            has_pdb = bool(data['pdb'])
            has_fold = bool(data['fold'])
            
            if has_pdb and has_fold:
                # Calculate averages
                avg_dock = (data['pdb']['dock_score'] + data['fold']['dock_score']) / 2
                avg_opt = (data['pdb']['opt_score'] + data['fold']['opt_score']) / 2
                
                # Write PDB result
                writer.writerow([
                    ligand_id, mutant, data['pdb']['receptor'], 'PDB',
                    f"{data['pdb']['dock_score']:.3f}",
                    f"{data['pdb']['opt_score']:.3f}",
                    f"{avg_dock:.3f}",
                    f"{avg_opt:.3f}"
                ])
                
                # Write AlphaFold result
                writer.writerow([
                    ligand_id, mutant, data['fold']['receptor'], 'AlphaFold',
                    f"{data['fold']['dock_score']:.3f}",
                    f"{data['fold']['opt_score']:.3f}",
                    f"{avg_dock:.3f}",
                    f"{avg_opt:.3f}"
                ])
            
            elif has_pdb:
                # Only PDB structure available
                writer.writerow([
                    ligand_id, mutant, data['pdb']['receptor'], 'PDB',
                    f"{data['pdb']['dock_score']:.3f}",
                    f"{data['pdb']['opt_score']:.3f}",
                    '', ''
                ])
            
            elif has_fold:
                # Only AlphaFold structure available
                writer.writerow([
                    ligand_id, mutant, data['fold']['receptor'], 'AlphaFold',
                    f"{data['fold']['dock_score']:.3f}",
                    f"{data['fold']['opt_score']:.3f}",
                    '', ''
                ])

print(f"\n✅ All docking complete. Results saved to {csv_file}")

#TODO
# 1. Improve protein preparation for more accurate docking scores
# 2. Expand PDB mutation proteins and get average scores
# 3. Use other docking tools and get average scores(Boltz/Gromacs/Schrodingers Glide)