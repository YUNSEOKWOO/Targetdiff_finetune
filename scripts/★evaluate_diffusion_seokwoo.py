import argparse
import os
import torch
import numpy as np
import pandas as pd  
from rdkit import Chem
from rdkit import RDLogger
from tqdm.auto import tqdm
from collections import Counter
from rdkit.Chem import rdMolDescriptors
from rdkit import DataStructs

from utils.evaluation import eval_atom_type, scoring_func, analyze, eval_bond_length
from utils import misc, reconstruct, transforms
from utils.evaluation.docking_qvina import QVinaDockingTask
from utils.evaluation.docking_vina import VinaDockingTask

from rdkit.Chem import Descriptors


def print_dict(d, logger):
    for k, v in d.items():
        if v is not None:
            logger.info(f'{k}:\t{v:.4f}')
        else:
            logger.info(f'{k}:\tNone')


def print_ring_ratio(all_ring_sizes, logger):
    for ring_size in range(3, 10):
        n_mol = 0
        for counter in all_ring_sizes:
            if ring_size in counter:
                n_mol += 1
        logger.info(f'ring size: {ring_size} ratio: {n_mol / len(all_ring_sizes):.3f}')


def calculate_validity(mols):
    """Calculate the validity of molecules."""
    valid_mols = []
    for mol in mols:
        try:
            Chem.SanitizeMol(mol)
            valid_mols.append(mol)
        except:
            continue
    return valid_mols, len(valid_mols) / len(mols)


def calculate_uniqueness(smiles_list):
    """Calculate the uniqueness of molecules."""
    unique_smiles = set(smiles_list)
    return len(unique_smiles) / len(smiles_list)


def calculate_novelty(smiles_list, known_smiles):
    """Calculate the novelty of molecules against known smiles."""
    novel_smiles = [sm for sm in smiles_list if sm not in known_smiles]
    return len(novel_smiles) / len(smiles_list)


def calculate_diversity(mols):
    """Calculate the diversity of molecules using Tanimoto similarity."""
    fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) for mol in mols]
    diversity = []
    for i in range(len(fps)):
        for j in range(i + 1, len(fps)):
            sim = DataStructs.TanimotoSimilarity(fps[i], fps[j])
            diversity.append(1 - sim)  # 1 - similarity is diversity
    return np.mean(diversity) if diversity else 0

def calculate_lipinski_properties(mol):
    """Calculate Lipinski's Rule of Five properties."""
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    return {'MW': mw, 'LogP': logp, 'HBD': hbd, 'HBA': hba}


def passes_lipinski(lipinski_props):
    """Check if the molecule passes Lipinski's Rule of Five."""
    return (lipinski_props['MW'] <= 500 and
            lipinski_props['LogP'] <= 5 and
            lipinski_props['HBD'] <= 5 and
            lipinski_props['HBA'] <= 10)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_path', type=str)
    parser.add_argument('--verbose', type=eval, default=False)
    parser.add_argument('--eval_step', type=int, default=-1)
    parser.add_argument('--eval_num_examples', type=int, default=None)
    parser.add_argument('--save', type=eval, default=True)
    parser.add_argument('--protein_root', type=str, default='./data/crossdocked_v1.1_rmsd1.0')
    parser.add_argument('--atom_enc_mode', type=str, default='add_aromatic')
    parser.add_argument('--docking_mode', type=str, choices=['qvina', 'vina_score', 'vina_dock', 'none'])
    parser.add_argument('--exhaustiveness', type=int, default=16)
    args = parser.parse_args()

    result_path = os.path.join(args.sample_path, 'eval_results')
    os.makedirs(result_path, exist_ok=True)
    logger = misc.get_logger('evaluate', log_dir=result_path)
    if not args.verbose:
        RDLogger.DisableLog('rdApp.*')

    # Load generated data from sample.pt
    sample_pt_path = os.path.join(args.sample_path, 'sample.pt')
    data = torch.load(sample_pt_path)

    mols = data['mols']  # list of rdkit.Chem.rdchem.Mol
    all_pred_ligand_pos_traj = data['pred_ligand_pos_traj']  # list of numpy arrays
    all_pred_ligand_v_traj = data['pred_ligand_v_traj']  # list of numpy arrays

    num_samples = len(mols)
    logger.info(f'Loaded {num_samples} molecules.')

    all_mol_stable, all_atom_stable, all_n_atom = 0, 0, 0
    n_recon_success, n_eval_success, n_complete = 0, 0, 0
    results = []
    all_pair_dist, all_bond_dist = [], []
    all_atom_types = Counter()
    success_pair_dist, success_atom_types = [], Counter()

    # Initialize list to collect per-molecule metrics
    per_molecule_metrics = []

    for idx in tqdm(range(num_samples), desc='Evaluating molecules'):
        mol = mols[idx]
        if mol is None:
            continue

        # Get the predicted trajectories for this molecule
        all_pred_ligand_pos = all_pred_ligand_pos_traj[idx]  # Shape: [num_steps, num_atoms, 3]
        all_pred_ligand_v = all_pred_ligand_v_traj[idx]      # Shape: [num_steps, num_atoms]

        if args.eval_step >= len(all_pred_ligand_pos):
            logger.warning(f'Evaluation step {args.eval_step} out of range for molecule {idx}')
            continue

        # Select the positions and atom types at the evaluation step
        pred_pos = all_pred_ligand_pos[args.eval_step]  # Shape: [num_atoms, 3]
        pred_v = all_pred_ligand_v[args.eval_step]      # Shape: [num_atoms]

        # Stability check
        pred_atom_type = transforms.get_atomic_number_from_index(pred_v, mode=args.atom_enc_mode)
        all_atom_types += Counter(pred_atom_type)
        r_stable = analyze.check_stability(pred_pos, pred_atom_type)
        all_mol_stable += r_stable[0]
        all_atom_stable += r_stable[1]
        all_n_atom += r_stable[2]

        pair_dist = eval_bond_length.pair_distance_from_pos_v(pred_pos, pred_atom_type)
        all_pair_dist += pair_dist

        # Reconstruction
        try:
            pred_aromatic = transforms.is_aromatic_from_index(pred_v, mode=args.atom_enc_mode)
            mol = reconstruct.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
            smiles = Chem.MolToSmiles(mol)
        except reconstruct.MolReconsError:
            if args.verbose:
                logger.warning(f'Reconstruct failed for molecule {idx}')
            continue
        n_recon_success += 1

        if '.' in smiles:
            continue
        n_complete += 1

        # Chemical and docking check
        try:
            # Perform chemical evaluation (QED, SA, etc.)
            chem_results = scoring_func.get_chem(mol)
            
            # Perform docking if required
            if args.docking_mode == 'qvina':
                # QVina docking task
                vina_task = QVinaDockingTask.from_generated_mol(ligand_rdmol=mol, ligand_filename=None, protein_root=args.protein_root)
                vina_results = vina_task.run_sync()

            elif args.docking_mode in ['vina_score', 'vina_dock']:
                # Vina docking task
                vina_task = VinaDockingTask.from_generated_mol(ligand_rdmol=mol, ligand_filename=None, protein_root=args.protein_root)
                # 'vina_score' mode: calculate score without docking
                score_only_results = vina_task.run(mode='score_only', exhaustiveness=args.exhaustiveness)
                # 'minimize' mode: perform energy minimization
                minimize_results = vina_task.run(mode='minimize', exhaustiveness=args.exhaustiveness)
                
                # Save results
                vina_results = {
                    'score_only': score_only_results,
                    'minimize': minimize_results
                }
                # 'vina_dock' mode: perform docking simulation
                if args.docking_mode == 'vina_dock':
                    docking_results = vina_task.run(mode='dock', exhaustiveness=args.exhaustiveness)
                    vina_results['dock'] = docking_results
            else:
                vina_results = None

            # Evaluation succeeded
            n_eval_success += 1

        except Exception as e:
            if args.verbose:
                logger.warning(f'Evaluation failed for molecule {idx}: {e}')
            continue

        # Now we only consider complete molecules as success
        bond_dist = eval_bond_length.bond_distance_from_mol(mol)
        all_bond_dist += bond_dist

        success_pair_dist += pair_dist
        success_atom_types += Counter(pred_atom_type)

        # Lipinski's Rule evaluation
        lipinski_props = calculate_lipinski_properties(mol)
        passes_lipinski_rule = passes_lipinski(lipinski_props)

        results.append({
            'mol': mol,
            'smiles': smiles,
            'pred_pos': pred_pos,
            'pred_v': pred_v,
            'chem_results': chem_results,
            'vina': vina_results
        })

        # Collect per-molecule metrics
        per_mol_data = {
            'index': idx,
            'smiles': smiles,
            'validity': 1,  # Valid since reconstruction and chemical checks passed
            'qed': chem_results['qed'],
            'sa': chem_results['sa'],
            'MW': lipinski_props['MW'],
            'LogP': lipinski_props['LogP'],
            'HBD': lipinski_props['HBD'],
            'HBA': lipinski_props['HBA'],
            'Lipinski': 1 if passes_lipinski_rule else 0,
        }

        # Get vina score depending on docking mode
        if args.docking_mode == 'qvina':
            per_mol_data['vina_score'] = vina_results[0]['affinity'] if vina_results else None
        elif args.docking_mode in ['vina_score', 'vina_dock']:
            per_mol_data['vina_score'] = vina_results['score_only'][0]['affinity'] if vina_results else None
        else:
            per_mol_data['vina_score'] = None

        per_molecule_metrics.append(per_mol_data)

    logger.info(f'Evaluation done! {num_samples} samples in total.')

    # Calculate overall validity metrics
    fraction_mol_stable = all_mol_stable / num_samples if num_samples > 0 else 0
    fraction_atm_stable = all_atom_stable / all_n_atom if all_n_atom > 0 else 0
    fraction_recon = n_recon_success / num_samples if num_samples > 0 else 0
    fraction_eval = n_eval_success / num_samples if num_samples > 0 else 0
    fraction_complete = n_complete / num_samples if num_samples > 0 else 0
    validity_dict = {
        'mol_stable': fraction_mol_stable,
        'atm_stable': fraction_atm_stable,
        'recon_success': fraction_recon,
        'eval_success': fraction_eval,
        'complete': fraction_complete
    }

    # Collect all valid molecules and their SMILES
    valid_mols = [r['mol'] for r in results]
    smiles_list = [r['smiles'] for r in results]

    # Calculate uniqueness per molecule
    smiles_counts = Counter(smiles_list)
    for per_mol_data in per_molecule_metrics:
        smi = per_mol_data['smiles']
        per_mol_data['uniqueness'] = 1 if smiles_counts[smi] == 1 else 0

    # Calculate novelty per molecule
    known_smiles = [...]  # Replace with known dataset SMILES strings
    known_smiles_set = set(known_smiles)
    for per_mol_data in per_molecule_metrics:
        smi = per_mol_data['smiles']
        per_mol_data['novelty'] = 0 if smi in known_smiles_set else 1

    # Calculate diversity per molecule
    fps = [rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) for mol in valid_mols]
    for idx, per_mol_data in enumerate(per_molecule_metrics):
        fp = fps[idx]
        similarities = []
        for j, fp_other in enumerate(fps):
            if idx != j:
                sim = DataStructs.TanimotoSimilarity(fp, fp_other)
                similarities.append(sim)
        if similarities:
            diversity = 1 - np.mean(similarities)  # Average dissimilarity
        else:
            diversity = 0
        per_mol_data['diversity'] = diversity

    # Calculate overall validity, uniqueness, novelty, and diversity
    validity = len(valid_mols) / len(mols) if len(mols) > 0 else 0
    uniqueness = calculate_uniqueness(smiles_list)
    novelty = calculate_novelty(smiles_list, known_smiles)
    diversity = calculate_diversity(valid_mols)

    # Print validity, uniqueness, novelty, and diversity
    logger.info(f'Validity: {validity:.4f}')
    logger.info(f'Uniqueness: {uniqueness:.4f}')
    logger.info(f'Novelty: {novelty:.4f}')
    logger.info(f'Diversity: {diversity:.4f}')

    # Lipinski's Rule compliance
    lipinski_passes = [pm['Lipinski'] for pm in per_molecule_metrics]
    lipinski_pass_rate = sum(lipinski_passes) / len(lipinski_passes) if len(lipinski_passes) > 0 else 0
    logger.info(f'Lipinski compliance: {lipinski_pass_rate:.4f}')

    # Average Lipinski properties
    avg_mw = np.mean([pm['MW'] for pm in per_molecule_metrics])
    avg_logp = np.mean([pm['LogP'] for pm in per_molecule_metrics])
    avg_hbd = np.mean([pm['HBD'] for pm in per_molecule_metrics])
    avg_hba = np.mean([pm['HBA'] for pm in per_molecule_metrics])
    logger.info(f'Average MW: {avg_mw:.2f}')
    logger.info(f'Average LogP: {avg_logp:.2f}')
    logger.info(f'Average HBD: {avg_hbd:.2f}')
    logger.info(f'Average HBA: {avg_hba:.2f}')

    # Bond length evaluation
    c_bond_length_profile = eval_bond_length.get_bond_length_profile(all_bond_dist)
    c_bond_length_dict = eval_bond_length.eval_bond_length_profile(c_bond_length_profile)
    logger.info('JS bond distances of complete molecules:')
    print_dict(c_bond_length_dict, logger)

    success_pair_length_profile = eval_bond_length.get_pair_length_profile(success_pair_dist)
    success_js_metrics = eval_bond_length.eval_pair_length_profile(success_pair_length_profile)
    print_dict(success_js_metrics, logger)

    atom_type_js = eval_atom_type.eval_atom_type_distribution(success_atom_types)
    logger.info('Atom type JS: %.4f' % atom_type_js)

    if args.save:
        eval_bond_length.plot_distance_hist(success_pair_length_profile,
                                            metrics=success_js_metrics,
                                            save_path=os.path.join(result_path, f'pair_dist_hist_{args.eval_step}.png'))

    logger.info('Number of reconstructed molecules: %d, complete molecules: %d, evaluated molecules: %d' % (
        n_recon_success, n_complete, len(results)))

    qed = [r['chem_results']['qed'] for r in results]
    sa = [r['chem_results']['sa'] for r in results]
    logger.info('QED:   Mean: %.3f Median: %.3f' % (np.mean(qed), np.median(qed)))
    logger.info('SA:    Mean: %.3f Median: %.3f' % (np.mean(sa), np.median(sa)))
    if args.docking_mode == 'qvina':
        vina = [r['vina'][0]['affinity'] for r in results]
        logger.info('Vina:  Mean: %.3f Median: %.3f' % (np.mean(vina), np.median(vina)))
    elif args.docking_mode in ['vina_dock', 'vina_score']:
        vina_score_only = [r['vina']['score_only'][0]['affinity'] for r in results]
        vina_min = [r['vina']['minimize'][0]['affinity'] for r in results]
        logger.info('Vina Score:  Mean: %.3f Median: %.3f' % (np.mean(vina_score_only), np.median(vina_score_only)))
        logger.info('Vina Min  :  Mean: %.3f Median: %.3f' % (np.mean(vina_min), np.median(vina_min)))
        if args.docking_mode == 'vina_dock':
            vina_dock = [r['vina']['dock'][0]['affinity'] for r in results]
            logger.info('Vina Dock :  Mean: %.3f Median: %.3f' % (np.mean(vina_dock), np.median(vina_dock)))

    # Check ring distribution
    print_ring_ratio([r['chem_results']['ring_size'] for r in results], logger)

    if args.save:
        torch.save({
            'stability': validity_dict,
            'bond_length': all_bond_dist,
            'all_results': results,
            'validity': validity,
            'uniqueness': uniqueness,
            'novelty': novelty,
            'diversity': diversity,
            'lipinski_compliance': lipinski_pass_rate  # Lipinski's Rule 준수율 저장
        }, os.path.join(result_path, f'metrics_{args.eval_step}.pt'))

        # Save per-molecule metrics to CSV
        df = pd.DataFrame(per_molecule_metrics)
        csv_save_path = os.path.join(result_path, f'per_molecule_metrics_{args.eval_step}.csv')
        df.to_csv(csv_save_path, index=False)
        logger.info(f'Per-molecule metrics saved to {csv_save_path}')
