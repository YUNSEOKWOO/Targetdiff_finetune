import argparse
import os
import shutil
import time

import numpy as np
import torch
from torch_geometric.data import Batch
from torch_geometric.transforms import Compose
from torch_scatter import scatter_sum, scatter_mean
from tqdm.auto import tqdm

import utils.misc as misc
import utils.transforms as trans
from datasets import get_dataset
from datasets.pl_data import FOLLOW_BATCH
from models.molopt_score_model_scaffold import ScorePosNet3D, log_sample_categorical
from utils.evaluation import atom_num


def unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms):
    all_step_v = [[] for _ in range(n_data)]
    for v in ligand_v_traj:  # step_i
        v_array = v.cpu().numpy()
        for k in range(n_data):
            all_step_v[k].append(v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
    all_step_v = [np.stack(step_v) for step_v in all_step_v]  # num_samples * [num_steps, num_atoms_i]
    return all_step_v

################################################################
def sample_diffusion_ligand(model, data, num_samples, batch_size=16, device='cuda:0',
                            num_steps=None, pos_only=False, center_pos_mode='protein',
                            sample_num_atoms='prior', is_scaffold_atom=None):
    # import time
    # from torch_scatter import scatter_sum
    # from torch_geometric.data import Batch
    # from utils.transforms import FOLLOW_BATCH  # FOLLOW_BATCH 변수를 임포트하거나 직접 정의합니다.

    all_pred_pos, all_pred_v = [], []
    all_pred_pos_traj, all_pred_v_traj = [], []
    all_pred_v0_traj, all_pred_vt_traj = [], []
    time_list = []
    num_batch = int(np.ceil(num_samples / batch_size))
    current_i = 0

    for i in tqdm(range(num_batch)):
        t1 = time.time()  # 시간 측정을 위해 t1 변수 정의
        n_data = batch_size if i < num_batch - 1 else num_samples - batch_size * (num_batch - 1)
        data_list = []
        for _ in range(n_data):
            data_clone = data.clone()
            data_clone.is_scaffold_atom = data.is_scaffold_atom.clone()
            data_list.append(data_clone)
        batch = Batch.from_data_list(data_list, follow_batch=FOLLOW_BATCH + ('is_scaffold_atom',)).to(device)
        
        with torch.no_grad():
            batch_protein = batch.protein_element_batch
            batch_ligand = batch.ligand_element_batch
            ligand_num_atoms = scatter_sum(torch.ones_like(batch_ligand), batch_ligand, dim=0).tolist()
            
            # 배치 내의 스캐폴드 마스크 생성
            is_scaffold_atom_batch = batch.is_scaffold_atom.bool().to(device)
            
            # 초기 리간드 위치 및 타입 설정
            init_ligand_pos = torch.randn_like(batch.ligand_pos)
            init_ligand_v = torch.randint(0, model.num_classes, (len(batch_ligand),), device=device)
            
            # 스캐폴드 원자의 위치와 타입은 데이터에서 가져옴
            scaffold_indices = is_scaffold_atom_batch.nonzero(as_tuple=False).view(-1)
            init_ligand_pos[scaffold_indices] = batch.ligand_pos[scaffold_indices]
            init_ligand_v[scaffold_indices] = batch.ligand_element[scaffold_indices]
            
            # 샘플링
            r = model.sample_diffusion(
                protein_pos=batch.protein_pos,
                protein_v=batch.protein_atom_feature.float(),
                batch_protein=batch_protein,

                init_ligand_pos=init_ligand_pos,
                init_ligand_v=init_ligand_v,
                batch_ligand=batch_ligand,
                num_steps=num_steps,
                pos_only=pos_only,
                center_pos_mode=center_pos_mode,
                is_scaffold_atom=is_scaffold_atom_batch
            )
            ligand_pos, ligand_v = r['pos'], r['v']
            ligand_pos_traj, ligand_v_traj = r['pos_traj'], r['v_traj']
            if not pos_only:
                ligand_v0_traj, ligand_vt_traj = r['v0_traj'], r['vt_traj']
            else:
                ligand_v0_traj, ligand_vt_traj = None, None

            # unbatch pos
            ligand_cum_atoms = np.cumsum([0] + ligand_num_atoms)
            ligand_pos_array = ligand_pos.cpu().numpy().astype(np.float64)
            all_pred_pos += [ligand_pos_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(n_data)]

            all_step_pos = [[] for _ in range(n_data)]
            for p in ligand_pos_traj:
                p_array = p.cpu().numpy().astype(np.float64)
                for k in range(n_data):
                    all_step_pos[k].append(p_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
            all_step_pos = [np.stack(step_pos) for step_pos in all_step_pos]
            all_pred_pos_traj += all_step_pos

            # unbatch v
            ligand_v_array = ligand_v.cpu().numpy()
            all_pred_v += [ligand_v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]] for k in range(n_data)]

            def unbatch_v_traj(v_traj, n_data, ligand_cum_atoms):
                all_step_v = [[] for _ in range(n_data)]
                for v in v_traj:
                    v_array = v.cpu().numpy()
                    for k in range(n_data):
                        all_step_v[k].append(v_array[ligand_cum_atoms[k]:ligand_cum_atoms[k + 1]])
                all_step_v = [np.stack(step_v) for step_v in all_step_v]
                return all_step_v

            all_step_v = unbatch_v_traj(ligand_v_traj, n_data, ligand_cum_atoms)
            all_pred_v_traj += all_step_v

            if not pos_only:
                all_step_v0 = unbatch_v_traj(ligand_v0_traj, n_data, ligand_cum_atoms)
                all_pred_v0_traj += all_step_v0
                all_step_vt = unbatch_v_traj(ligand_vt_traj, n_data, ligand_cum_atoms)
                all_pred_vt_traj += all_step_vt

        t2 = time.time()
        time_list.append(t2 - t1)
        current_i += n_data

    return all_pred_pos, all_pred_v, all_pred_pos_traj, all_pred_v_traj, \
           all_pred_v0_traj, all_pred_vt_traj, time_list



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str)
    parser.add_argument('-i', '--data_id', type=int)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--result_path', type=str, default='./outputs')
    args = parser.parse_args()

    logger = misc.get_logger('sampling')

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
        ligand_featurizer,
        trans.FeaturizeLigandBond(),
    ])

    # Load dataset
    dataset, subsets = get_dataset(
        config=ckpt['config'].data,
        transform=transform
    )
    train_set, test_set = subsets['train'], subsets['test']
    logger.info(f'Successfully load the dataset (size: {len(test_set)})!')

    # Load model
    model = ScorePosNet3D(
        ckpt['config'].model,
        protein_atom_feature_dim=protein_featurizer.feature_dim,
        ligand_atom_feature_dim=ligand_featurizer.feature_dim
    ).to(args.device)
    model.load_state_dict(ckpt['model'])
    logger.info(f'Successfully load the model! {config.model.checkpoint}')

    data = test_set[args.data_id]
    pred_pos, pred_v, pred_pos_traj, pred_v_traj, pred_v0_traj, pred_vt_traj, time_list = sample_diffusion_ligand(
        model, data, config.sample.num_samples,
        batch_size=args.batch_size, device=args.device,
        num_steps=config.sample.num_steps,
        pos_only=config.sample.pos_only,
        center_pos_mode=config.sample.center_pos_mode,
        sample_num_atoms=config.sample.sample_num_atoms
    )
    result = {
        'data': data,
        'pred_ligand_pos': pred_pos,
        'pred_ligand_v': pred_v,
        'pred_ligand_pos_traj': pred_pos_traj,
        'pred_ligand_v_traj': pred_v_traj,
        'time': time_list
    }
    logger.info('Sample done!')

    result_path = args.result_path
    os.makedirs(result_path, exist_ok=True)
    shutil.copyfile(args.config, os.path.join(result_path, 'sample.yml'))
    torch.save(result, os.path.join(result_path, f'result_{args.data_id}.pt'))
