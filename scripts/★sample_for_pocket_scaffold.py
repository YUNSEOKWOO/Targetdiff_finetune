import argparse
import os
import shutil

import torch
from torch_geometric.transforms import Compose

import utils.misc as misc
import utils.transforms as trans
from datasets.pl_data import ProteinLigandData, torchify_dict
from models.molopt_score_model_scaffold import ScorePosNet3D
from scripts.sample_diffusion_scaffold import sample_diffusion_ligand
from utils.data import PDBProtein
from utils import reconstruct_scaffold
from rdkit import Chem

###############################################################
from rdkit.Chem.Scaffolds import MurckoScaffold
def pdb_to_pocket_data(pdb_path):
    # PDB 파일에서 단백질과 리간드 정보를 추출합니다.
    protein = PDBProtein(pdb_path)
    pocket_dict = protein.to_dict_atom()
    
    # 리간드 추출
    ligand_mol = protein.get_ligand_mol()
    if ligand_mol is None:
        raise ValueError("PDB 파일에서 리간드를 추출할 수 없습니다.")
    
    # 스캐폴드 추출
    scaffold_mol = MurckoScaffold.GetScaffoldForMol(ligand_mol)
    if scaffold_mol is None:
        raise ValueError("리간드에서 스캐폴드를 추출할 수 없습니다.")
    
    # 스캐폴드와 비스캐폴드 원자를 구분
    scaffold_atom_indices = scaffold_mol.GetSubstructMatch(scaffold_mol)
    total_atom_indices = list(range(ligand_mol.GetNumAtoms()))
    non_scaffold_atom_indices = [idx for idx in total_atom_indices if idx not in scaffold_atom_indices]
    
    # 원자 특징과 위치 추출
    atom_types = []
    positions = []
    is_scaffold_atom = []
    for atom_idx in total_atom_indices:
        atom = ligand_mol.GetAtomWithIdx(atom_idx)
        atom_types.append(atom.GetAtomicNum())
        pos = ligand_mol.GetConformer().GetAtomPosition(atom_idx)
        positions.append([pos.x, pos.y, pos.z])
        is_scaffold_atom.append(atom_idx in scaffold_atom_indices)
    
    # 결합 정보 추출
    bond_index = []
    bond_type = []
    for bond in ligand_mol.GetBonds():
        begin_idx = bond.GetBeginAtomIdx()
        end_idx = bond.GetEndAtomIdx()
        bond_index.append([begin_idx, end_idx])
        bond_type.append(bond.GetBondTypeAsDouble())
    
    ligand_dict = {
        'element': torch.tensor(atom_types, dtype=torch.long),
        'pos': torch.tensor(positions, dtype=torch.float),
        'bond_index': torch.tensor(bond_index, dtype=torch.long).t().contiguous(),
        'bond_type': torch.tensor(bond_type, dtype=torch.long),
        # 추가적인 원자 특징이 있으면 포함시킵니다.
    }
    
    data = ProteinLigandData.from_protein_ligand_dicts(
        protein_dict=torchify_dict(pocket_dict),
        ligand_dict=torchify_dict(ligand_dict)
    )
    data.is_scaffold_atom = torch.tensor(is_scaffold_atom, dtype=torch.bool)
    
    return data
###############################################################

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('--pdb_path', type=str)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--result_path', type=str, default='./outputs_pdb')
    parser.add_argument('--num_samples', type=int)
    args = parser.parse_args()

    logger = misc.get_logger('evaluate')

    # Load config
    config = misc.load_config(args.config)
    logger.info(config)
    misc.seed_all(config.sample.seed)

    # Load checkpoint
    ckpt = torch.load(config.model.checkpoint, map_location=args.device)
    logger.info(f"Training Config: {ckpt['config']}")

    # Transforms
    protein_featurizer = trans.FeaturizeProteinAtom()
    ligand_atom_mode = ckpt['config'].data.transform.ligand_atom_mode
    ligand_featurizer = trans.FeaturizeLigandAtom(ligand_atom_mode)
    transform = Compose([
        protein_featurizer,
    ])

    # Load model
    model = ScorePosNet3D(
        ckpt['config'].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)
    model.load_state_dict(ckpt['model'], strict=False if 'train_config' in config.model else True)
    logger.info(f'Successfully load the model! {config.model.checkpoint}')

###############################################################
    # Load pocket
    data = pdb_to_pocket_data(args.pdb_path)
    data = transform(data)
    if args.num_samples:
        config.sample.num_samples = args.num_samples

    # 스캐폴드 마스크를 추출
    is_scaffold_atom = data.is_scaffold_atom.to(args.device)
    
    # 샘플링
    all_pred_pos, all_pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list = sample_diffusion_ligand(
        model, data, config.sample.num_samples,
        batch_size=args.batch_size, device=args.device,
        num_steps=config.sample.num_steps,
        pos_only=config.sample.pos_only,
        center_pos_mode=config.sample.center_pos_mode,
        sample_num_atoms=config.sample.sample_num_atoms,
        is_scaffold_atom=is_scaffold_atom
    )
###############################################################
    result = {
        'data': data,
        'pred_ligand_pos': all_pred_pos,
        'pred_ligand_v': all_pred_v,
        'pred_ligand_pos_traj': pred_pos_traj,
        'pred_ligand_v_traj': pred_v_traj
    }
    # print('pred_ligand', result)
    logger.info('Sample done!')

    # reconstruction
    gen_mols = []
    n_recon_success, n_complete = 0, 0
    for sample_idx, (pred_pos, pred_v) in enumerate(zip(all_pred_pos, all_pred_v)):
        pred_atom_type = trans.get_atomic_number_from_index(pred_v, mode='add_aromatic')
        try:
            pred_aromatic = trans.is_aromatic_from_index(pred_v, mode='add_aromatic')
            mol = reconstruct_scaffold.reconstruct_from_generated(pred_pos, pred_atom_type, pred_aromatic)
            smiles = Chem.MolToSmiles(mol)
        except reconstruct_scaffold.MolReconsError:
            gen_mols.append(None)
            continue
        n_recon_success += 1

        if '.' in smiles:
            gen_mols.append(None)
            continue
        n_complete += 1
        gen_mols.append(mol)
    result['mols'] = gen_mols
    logger.info('Reconstruction done!')
    logger.info(f'n recon: {n_recon_success} n complete: {n_complete}')

    result_path = args.result_path
    os.makedirs(result_path, exist_ok=True)
    shutil.copyfile(args.config, os.path.join(result_path, 'sample.yml'))
    torch.save(result, os.path.join(result_path, f'sample.pt'))
    mols_save_path = os.path.join(result_path, f'sdf')
    os.makedirs(mols_save_path, exist_ok=True)
    for idx, mol in enumerate(gen_mols):
        if mol is not None:
            sdf_writer = Chem.SDWriter(os.path.join(mols_save_path, f'{idx:03d}.sdf'))
            sdf_writer.write(mol)
            sdf_writer.close()
    logger.info(f'Results are saved in {result_path}')
