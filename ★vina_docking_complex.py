########pdb형식의 리간드 파일로 도킹하는 코드########

from openbabel import pybel
from meeko import MoleculePreparation
from meeko import obutils
from vina import Vina
import subprocess
import rdkit.Chem as Chem
from rdkit.Chem import AllChem
import tempfile
import AutoDockTools
import os
import contextlib
import lmdb
import os
import pandas as pd

from utils.reconstruct import reconstruct_from_generated
from utils.evaluation.docking_qvina import get_random_id, BaseDockingTask


def supress_stdout(func):
    def wrapper(*a, **ka):
        with open(os.devnull, 'w') as devnull:
            with contextlib.redirect_stdout(devnull):
                return func(*a, **ka)
    return wrapper


class PrepLig(object):
    def __init__(self, input_mol, mol_format):
        if mol_format == 'smi':
            self.ob_mol = pybel.readstring('smi', input_mol)
        elif mol_format in ['sdf', 'pdb']: 
            self.ob_mol = next(pybel.readfile(mol_format, input_mol))
        else:
            raise ValueError(f'mol_format {mol_format} not supported')
        
    def addH(self, polaronly=False, correctforph=True, PH=7): 
        self.ob_mol.OBMol.AddHydrogens(polaronly, correctforph, PH)
        obutils.writeMolecule(self.ob_mol.OBMol, 'tmp_h.sdf')

    def gen_conf(self):
        # 분자가 이미 3D 좌표를 가지고 있는지 확인
        if self.ob_mol.dim == 3:
            print("분자가 이미 3D 좌표를 가지고 있습니다. 컨포머 생성을 건너뜁니다.")
        else:
            # 3D 좌표가 없는 경우 컨포머 생성
            sdf_block = self.ob_mol.write('sdf')
            rdkit_mol = Chem.MolFromMolBlock(sdf_block, removeHs=False)
            AllChem.EmbedMolecule(rdkit_mol, Chem.rdDistGeom.ETKDGv3())
            self.ob_mol = pybel.readstring('sdf', Chem.MolToMolBlock(rdkit_mol))
            obutils.writeMolecule(self.ob_mol.OBMol, 'conf_h.sdf')
    # def gen_conf(self):
    #     sdf_block = self.ob_mol.write('sdf')
    #     rdkit_mol = Chem.MolFromMolBlock(sdf_block, removeHs=False)
    #     AllChem.EmbedMolecule(rdkit_mol, Chem.rdDistGeom.ETKDGv3())
    #     self.ob_mol = pybel.readstring('sdf', Chem.MolToMolBlock(rdkit_mol))
    #     obutils.writeMolecule(self.ob_mol.OBMol, 'conf_h.sdf')

    @supress_stdout
    def get_pdbqt(self, lig_pdbqt=None):
        preparator = MoleculePreparation()
        preparator.prepare(self.ob_mol.OBMol)
        if lig_pdbqt is not None: 
            preparator.write_pdbqt_file(lig_pdbqt)
            return 
        else: 
            return preparator.write_pdbqt_string()


class PrepProt(object): 
    def __init__(self, pdb_file): 
        self.prot = pdb_file
    
    def del_water(self, dry_pdb_file): # optional
        with open(self.prot) as f: 
            lines = [l for l in f.readlines() if l.startswith('ATOM') or l.startswith('HETATM')] 
            dry_lines = [l for l in lines if not 'HOH' in l]
        
        with open(dry_pdb_file, 'w') as f:
            f.write(''.join(dry_lines))
        self.prot = dry_pdb_file
        
    def addH(self, prot_pqr):  # call pdb2pqr
        self.prot_pqr = prot_pqr
        subprocess.Popen(['pdb2pqr30','--ff=AMBER',self.prot, self.prot_pqr],
                         stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL).communicate()

    def get_pdbqt(self, prot_pdbqt):
        prepare_receptor = os.path.join(AutoDockTools.__path__[0], 'Utilities24/prepare_receptor4.py')
        subprocess.Popen(['python3', prepare_receptor, '-r', self.prot_pqr, '-o', prot_pdbqt],
                         stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL).communicate()


class VinaDock(object): 
    def __init__(self, lig_pdbqt, prot_pdbqt): 
        self.lig_pdbqt = lig_pdbqt
        self.prot_pdbqt = prot_pdbqt
    
    def _max_min_pdb(self, pdb, buffer):
        with open(pdb, 'r') as f: 
            lines = [l for l in f.readlines() if l.startswith('ATOM') or l.startswith('HEATATM')]
            xs = [float(l[31:39]) for l in lines]
            ys = [float(l[39:47]) for l in lines]
            zs = [float(l[47:55]) for l in lines]
            print(max(xs), min(xs))
            print(max(ys), min(ys))
            print(max(zs), min(zs))
            pocket_center = [(max(xs) + min(xs))/2, (max(ys) + min(ys))/2, (max(zs) + min(zs))/2]
            box_size = [(max(xs) - min(xs)) + buffer, (max(ys) - min(ys)) + buffer, (max(zs) - min(zs)) + buffer]
            return pocket_center, box_size
    
    def get_box(self, center=None, box_size=None, ref=None, buffer=0):
        '''
        center: list or tuple of x, y, z coordinates for the pocket center
        box_size: list or tuple of x, y, z dimensions for the box size
        ref: reference pdb to define pocket. 
        buffer: buffer size to add 

        if center and box_size are provided:
            use them directly
        elif ref is not None:
            get the max and min on x, y, z axis in ref pdb and add buffer to each dimension 
        else: 
            use the entire protein to define pocket 
        '''
        if center is not None and box_size is not None:
            self.pocket_center = center
            self.box_size = box_size
        else:
            if ref is None: 
                ref = self.prot_pdbqt
            self.pocket_center, self.box_size = self._max_min_pdb(ref, buffer)
        print('Pocket center:', self.pocket_center)
        print('Box size:', self.box_size)


    def dock(self, score_func='vina', seed=0, mode='dock', exhaustiveness=16, save_pose=False, **kwargs):  # seed=0 mean random seed
        v = Vina(sf_name=score_func, seed=seed, verbosity=0, **kwargs)
        v.set_receptor(self.prot_pdbqt)
        v.set_ligand_from_file(self.lig_pdbqt)
        v.compute_vina_maps(center=self.pocket_center, box_size=self.box_size)
        if mode == 'score_only': 
            score = v.score()[0]
        elif mode == 'minimize':
            score = v.optimize()[0]
        elif mode == 'dock':
            v.dock(exhaustiveness=exhaustiveness, n_poses=4)
            score = v.energies(n_poses=4)[0][0]
        else:
            raise ValueError
        
        if not save_pose: 
            return score
        else: 
            if mode == 'score_only': 
                pose = None 
            elif mode == 'minimize': 
                tmp = tempfile.NamedTemporaryFile()
                with open(tmp.name, 'w') as f: 
                    v.write_pose(tmp.name, overwrite=True)             
                with open(tmp.name, 'r') as f: 
                    pose = f.read()
   
            elif mode == 'dock': 
                pose = v.poses(n_poses=4)
            else:
                raise ValueError
            return score, pose


def main():
    #단백질 준비
    # protein_file = 'data/mpro_complex/pocket/7uup_B_4WI_pocket10.pdb' # 'data/target_pdb/mpro_6y2e.pdb'
    # protein_pdbqt = 'data/mpro_complex/pocket/7uup_B_4WI_pocket10.pdbqt' # 'data/target_pdb/mpro_6y2e.pdbqt'
    # protein_pqr = 'data/mpro_complex/pocket/7uup_B_4WI_pocket10.pqr' # 'data/target_pdb/mpro_6y2e.pqr'
    # protein_file = 'data/target_pdb/7l11_A_protein.pdb'
    # protein_pdbqt = 'data/target_pdb/7l11_A_protein.pdbqt'
    # protein_pqr = 'data/target_pdb/7l11_A_protein.pqr'
    protein_file = 'data/mpro_complex/pocket/7l11_C_XF1_pocket10.pdb'
    protein_pdbqt = 'data/mpro_complex/pocket/7l11_C_XF1_pocket10.pdbqt'
    protein_pqr = 'data/mpro_complex/pocket/7l11_C_XF1_pocket10.pqr'

    # # 단백질 준비 단계
    prot = PrepProt(protein_file)
    prot.addH(protein_pqr)
    prot.get_pdbqt(protein_pdbqt)

    # 리간드 파일들이 저장된 디렉토리
    # ligands_dir = 'data/ligands_pdb' 
    # ligands_pdbqt_dir = 'data/ligands_pdbqt'  
    # docked_ligands_dir = 'data/docked_ligands_7l11_autodock/pocket10'
    ligands_dir = 'data/check_error' 
    ligands_pdbqt_dir = 'data/check_error/ligands_pdbqt'  
    docked_ligands_dir = 'data/check_error/docked_result'
    # ligands_dir = 'outputs_pdb/drugGPT_VAE/druggpt_finetuned_100_pdb'
    # ligands_pdbqt_dir = 'outputs_pdb/drugGPT_VAE/druggpt_finetuned_100_pdbqt'
    # docked_ligands_dir = 'outputs_pdb/drugGPT_VAE/druggpt_finetuned_100_docked_results'

    # ligands_dir = 'outputs_pdb/drugGPT_VAE/druggpt_normal_100_pdb'
    # ligands_pdbqt_dir = 'outputs_pdb/drugGPT_VAE/druggpt_normal_100_pdbqt'
    # docked_ligands_dir = 'outputs_pdb/drugGPT_VAE/druggpt_normal_100_docked_results'

    # ligands_dir = 'outputs_pdb/drugGPT_VAE/vae_finetuned_100_pdb'
    # ligands_pdbqt_dir = 'outputs_pdb/drugGPT_VAE/vae_finetuned_100_pdbqt'
    # docked_ligands_dir = 'outputs_pdb/drugGPT_VAE/vae_finetuned_100_docked_results'

    # ligands_pdbqt 디렉토리가 없으면 생성
    if not os.path.exists(ligands_pdbqt_dir):
        os.makedirs(ligands_pdbqt_dir)
    # docked_ligands 디렉토리가 없으면 생성
    if not os.path.exists(docked_ligands_dir):
        os.makedirs(docked_ligands_dir)

    # List ligand files
    ligand_files = sorted(os.listdir(ligands_dir))
    
    # Specify the center coordinates and box size for docking
    # Replace these values with your desired coordinates and box dimensions
    docking_center = [-21.814814, -4.2160625, -27.983782]  # Replace with actual values
    docking_box_size = [30.0, 30.0, 30.0]   # Replace with actual values
    # docking_center = None
    # docking_box_size = None
    
    for ligand_file in ligand_files:
        if ligand_file.endswith('.pdb'):
            try:
                ligand_path = os.path.join(ligands_dir, ligand_file)
                ligand_pdbqt = os.path.join(ligands_pdbqt_dir, ligand_file.replace('.pdb', '.pdbqt'))
                # ligand_pdbqt = ligand_file.replace('.pdb', '.pdbqt')

                # 리간드 준비
                lig = PrepLig(ligand_path, 'pdb')
                lig.get_pdbqt(ligand_pdbqt)

                # 도킹 설정 및 실행
                dock = VinaDock(ligand_pdbqt, protein_pdbqt)

                # Set docking box using specified center coordinates and box size
                dock.get_box(center=docking_center, box_size=docking_box_size)

                # 도킹 수행
                score, pose = dock.dock(score_func='vina', seed=0, mode='dock', exhaustiveness=16, save_pose=True)

                # 도킹 결과 저장
                docked_ligand_pdbqt = f'{ligand_file.replace(".pdb", "")}_score_{score:.2f}.pdbqt'
                docked_ligand_path = os.path.join(docked_ligands_dir, docked_ligand_pdbqt)

                with open(docked_ligand_path, 'w') as f:
                    f.write(pose)

                print(f'{ligand_file} 도킹 완료: 스코어 = {score} kcal/mol')
                print(f'도킹 결과가 {docked_ligand_path} 파일에 저장되었습니다.')

            except Exception as e:
                print(f'{ligand_file} 처리 중 오류 발생: {str(e)}. 파일을 건너뜁니다.')



if __name__ == '__main__':
    main()



##########도킹 실패한 것들에 대해서만 도킹 다시하는 것################
# def missing_ligands_docking():
#     # 단백질 준비
#     protein_file = 'data/target_pdb/7l11_A_protein.pdb'
#     protein_pdbqt = 'data/target_pdb/7l11_A_protein.pdbqt'
#     protein_pqr = 'data/target_pdb/7l11_A_protein.pqr'

#     # 단백질 준비 단계
#     prot = PrepProt(protein_file)
#     prot.addH(protein_pqr)
#     prot.get_pdbqt(protein_pdbqt)

#     # 리간드 파일들이 저장된 디렉토리
#     ligands_dir = 'data/ligands_pdb'
#     ligands_pdbqt_dir = 'data/ligands_pdbqt'
#     docked_ligands_dir = 'data/docked_ligands_7l11_autodock'

#     # ligands_pdbqt 디렉토리가 없으면 생성
#     if not os.path.exists(ligands_pdbqt_dir):
#         os.makedirs(ligands_pdbqt_dir)
#     # docked_ligands 디렉토리가 없으면 생성
#     if not os.path.exists(docked_ligands_dir):
#         os.makedirs(docked_ligands_dir)

#     # missing_index가 포함된 CSV 파일 로드
#     missing_index_file = 'missing_ligands.csv'  # CSV 파일 경로를 지정
#     missing_index_df = pd.read_csv(missing_index_file)

#     # missing_index에 해당하는 리간드 파일 리스트 가져오기
#     missing_index_list = missing_index_df['missing_index'].astype(str).tolist()

#     # List ligand files in ligands_dir and filter by missing_index
#     ligand_files = sorted(os.listdir(ligands_dir))
    
#     # Specify the center coordinates and box size for docking
#     docking_center = [-21.814814, -4.2160625, -27.983782]  # Replace with actual values
#     docking_box_size = [10.0, 10.0, 10.0]   # Replace with actual values
    
#     for ligand_file in ligand_files:
#         ligand_base = ligand_file.split('_')[0]  # 파일명의 첫 번째 부분을 기준으로 index 추출
#         if ligand_base in missing_index_list and ligand_file.endswith('.pdb'):
#             try:
#                 ligand_path = os.path.join(ligands_dir, ligand_file)
#                 ligand_pdbqt = os.path.join(ligands_pdbqt_dir, ligand_file.replace('.pdb', '.pdbqt'))

#                 # 리간드 준비
#                 lig = PrepLig(ligand_path, 'pdb')
#                 lig.get_pdbqt(ligand_pdbqt)

#                 # 도킹 설정 및 실행
#                 dock = VinaDock(ligand_pdbqt, protein_pdbqt)

#                 # Set docking box using specified center coordinates and box size
#                 dock.get_box(center=docking_center, box_size=docking_box_size)

#                 # 도킹 수행
#                 score, pose = dock.dock(score_func='vina', seed=0, mode='dock', exhaustiveness=8, save_pose=True)

#                 # 도킹 결과 저장
#                 docked_ligand_pdbqt = f'{ligand_file.replace(".pdb", "")}_score_{score:.2f}.pdbqt'
#                 docked_ligand_path = os.path.join(docked_ligands_dir, docked_ligand_pdbqt)

#                 with open(docked_ligand_path, 'w') as f:
#                     f.write(pose)

#                 print(f'{ligand_file} 도킹 완료: 스코어 = {score} kcal/mol')
#                 print(f'도킹 결과가 {docked_ligand_path} 파일에 저장되었습니다.')

#             except Exception as e:
#                 print(f'{ligand_file} 처리 중 오류 발생: {str(e)}. 파일을 건너뜁니다.')

# missing_ligands_docking()



# #############sdf파일로 도킹하는 코드################
# from openbabel import pybel
# from meeko import MoleculePreparation
# from meeko import obutils
# from vina import Vina
# import subprocess
# import rdkit.Chem as Chem
# from rdkit.Chem import AllChem
# import tempfile
# import AutoDockTools
# import os
# import contextlib
# import lmdb

# from utils.reconstruct import reconstruct_from_generated
# from utils.evaluation.docking_qvina import get_random_id, BaseDockingTask


# def supress_stdout(func):
#     def wrapper(*a, **ka):
#         with open(os.devnull, 'w') as devnull:
#             with contextlib.redirect_stdout(devnull):
#                 return func(*a, **ka)
#     return wrapper


# class PrepLig(object):
#     def __init__(self, input_mol, mol_format):
#         if mol_format == 'smi':
#             self.ob_mol = pybel.readstring('smi', input_mol)
#         elif mol_format in ['sdf', 'pdb']: 
#             self.ob_mol = next(pybel.readfile(mol_format, input_mol))
#         else:
#             raise ValueError(f'mol_format {mol_format} not supported')
        
#     def addH(self, polaronly=False, correctforph=True, PH=7): 
#         self.ob_mol.OBMol.AddHydrogens(polaronly, correctforph, PH)
#         obutils.writeMolecule(self.ob_mol.OBMol, 'tmp_h.sdf')

#     def gen_conf(self):
#         # 분자가 이미 3D 좌표를 가지고 있는지 확인
#         if self.ob_mol.dim == 3:
#             print("분자가 이미 3D 좌표를 가지고 있습니다. 컨포머 생성을 건너뜁니다.")
#         else:
#             # 3D 좌표가 없는 경우 컨포머 생성
#             sdf_block = self.ob_mol.write('sdf')
#             rdkit_mol = Chem.MolFromMolBlock(sdf_block, removeHs=False)
#             AllChem.EmbedMolecule(rdkit_mol, Chem.rdDistGeom.ETKDGv3())
#             self.ob_mol = pybel.readstring('sdf', Chem.MolToMolBlock(rdkit_mol))
#             obutils.writeMolecule(self.ob_mol.OBMol, 'conf_h.sdf')
#     # def gen_conf(self):
#     #     sdf_block = self.ob_mol.write('sdf')
#     #     rdkit_mol = Chem.MolFromMolBlock(sdf_block, removeHs=False)
#     #     AllChem.EmbedMolecule(rdkit_mol, Chem.rdDistGeom.ETKDGv3())
#     #     self.ob_mol = pybel.readstring('sdf', Chem.MolToMolBlock(rdkit_mol))
#     #     obutils.writeMolecule(self.ob_mol.OBMol, 'conf_h.sdf')

#     @supress_stdout
#     def get_pdbqt(self, lig_pdbqt=None):
#         preparator = MoleculePreparation()
#         preparator.prepare(self.ob_mol.OBMol)
#         if lig_pdbqt is not None: 
#             preparator.write_pdbqt_file(lig_pdbqt)
#             return 
#         else: 
#             return preparator.write_pdbqt_string()


# class PrepProt(object): 
#     def __init__(self, pdb_file): 
#         self.prot = pdb_file
    
#     def del_water(self, dry_pdb_file): # optional
#         with open(self.prot) as f: 
#             lines = [l for l in f.readlines() if l.startswith('ATOM') or l.startswith('HETATM')] 
#             dry_lines = [l for l in lines if not 'HOH' in l]
        
#         with open(dry_pdb_file, 'w') as f:
#             f.write(''.join(dry_lines))
#         self.prot = dry_pdb_file
        
#     def addH(self, prot_pqr):  # call pdb2pqr
#         self.prot_pqr = prot_pqr
#         subprocess.Popen(['pdb2pqr30','--ff=AMBER',self.prot, self.prot_pqr],
#                          stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL).communicate()

#     def get_pdbqt(self, prot_pdbqt):
#         prepare_receptor = os.path.join(AutoDockTools.__path__[0], 'Utilities24/prepare_receptor4.py')
#         subprocess.Popen(['python3', prepare_receptor, '-r', self.prot_pqr, '-o', prot_pdbqt],
#                          stderr=subprocess.DEVNULL, stdout=subprocess.DEVNULL).communicate()


# class VinaDock(object): 
#     def __init__(self, lig_pdbqt, prot_pdbqt): 
#         self.lig_pdbqt = lig_pdbqt
#         self.prot_pdbqt = prot_pdbqt
    
#     def _max_min_pdb(self, pdb, buffer):
#         with open(pdb, 'r') as f: 
#             lines = [l for l in f.readlines() if l.startswith('ATOM') or l.startswith('HEATATM')]
#             xs = [float(l[31:39]) for l in lines]
#             ys = [float(l[39:47]) for l in lines]
#             zs = [float(l[47:55]) for l in lines]
#             print(max(xs), min(xs))
#             print(max(ys), min(ys))
#             print(max(zs), min(zs))
#             pocket_center = [(max(xs) + min(xs))/2, (max(ys) + min(ys))/2, (max(zs) + min(zs))/2]
#             box_size = [(max(xs) - min(xs)) + buffer, (max(ys) - min(ys)) + buffer, (max(zs) - min(zs)) + buffer]
#             return pocket_center, box_size
    
#     def get_box(self, center=None, box_size=None, ref=None, buffer=0):
#         '''
#         center: list or tuple of x, y, z coordinates for the pocket center
#         box_size: list or tuple of x, y, z dimensions for the box size
#         ref: reference pdb to define pocket. 
#         buffer: buffer size to add 

#         if center and box_size are provided:
#             use them directly
#         elif ref is not None:
#             get the max and min on x, y, z axis in ref pdb and add buffer to each dimension 
#         else: 
#             use the entire protein to define pocket 
#         '''
#         if center is not None and box_size is not None:
#             self.pocket_center = center
#             self.box_size = box_size
#         else:
#             if ref is None: 
#                 ref = self.prot_pdbqt
#             self.pocket_center, self.box_size = self._max_min_pdb(ref, buffer)
#         print('Pocket center:', self.pocket_center)
#         print('Box size:', self.box_size)

#     def dock(self, score_func='vina', seed=0, mode='dock', exhaustiveness=8, save_pose=False, **kwargs):  # seed=0 mean random seed
#         v = Vina(sf_name=score_func, seed=seed, verbosity=0, **kwargs)
#         v.set_receptor(self.prot_pdbqt)
#         v.set_ligand_from_file(self.lig_pdbqt)
#         v.compute_vina_maps(center=self.pocket_center, box_size=self.box_size)
#         if mode == 'score_only': 
#             score = v.score()[0]
#         elif mode == 'minimize':
#             score = v.optimize()[0]
#         elif mode == 'dock':
#             v.dock(exhaustiveness=exhaustiveness, n_poses=1)
#             score = v.energies(n_poses=1)[0][0]
#         else:
#             raise ValueError
        
#         if not save_pose: 
#             return score
#         else: 
#             if mode == 'score_only': 
#                 pose = None 
#             elif mode == 'minimize': 
#                 tmp = tempfile.NamedTemporaryFile()
#                 with open(tmp.name, 'w') as f: 
#                     v.write_pose(tmp.name, overwrite=True)             
#                 with open(tmp.name, 'r') as f: 
#                     pose = f.read()
   
#             elif mode == 'dock': 
#                 pose = v.poses(n_poses=1)
#             else:
#                 raise ValueError
#             return score, pose


# def main():

#     # docking_center = [-21.814814, -4.2160625, -27.983782]  # Replace with actual values
#     docking_box_size = [10.0, 10.0, 10.0]   # Replace with actual values
#     docking_center = [-17.0685, 16.3875, -25.832]  # Replace with actual values
#     # docking_box_size = None   # Replace with actual values
#     # protein_file = 'data/target_pdb/7l11_A_protein.pdb' # 'data/target_pdb/mpro_6y2e.pdb'
#     # protein_pdbqt = 'data/target_pdb/7l11_A_protein.pdbqt' # 'data/target_pdb/mpro_6y2e.pdbqt'
#     # protein_pqr = 'data/target_pdb/7l11_A_protein.pqr' # 'data/target_pdb/mpro_6y2e.pqr'
#     protein_file = 'data/mpro_complex/pocket/7uup_B_4WI_pocket10.pdb' # 'data/target_pdb/mpro_6y2e.pdb'
#     protein_pdbqt = 'data/mpro_complex/pocket/7uup_B_4WI_pocket10.pdbqt' # 'data/target_pdb/mpro_6y2e.pdbqt'
#     protein_pqr = 'data/mpro_complex/pocket/7uup_B_4WI_pocket10.pqr' # 'data/target_pdb/mpro_6y2e.pqr'
#     # protein_file = 'data/target_pdb/7uup.pdb' # 'data/target_pdb/mpro_6y2e.pdb'
#     # protein_pdbqt = 'data/target_pdb/7uup.pdbqt' # 'data/target_pdb/mpro_6y2e.pdbqt'
#     # protein_pqr = 'data/target_pdb/7uup.pqr' # 'data/target_pdb/mpro_6y2e.pqr'

#     # # 단백질 준비 단계
#     prot = PrepProt(protein_file)
#     prot.addH(protein_pqr)
#     prot.get_pdbqt(protein_pdbqt)

#     # 리간드 파일들이 저장된 디렉토리
#     # ligands_dir = 'data/ligands_pdb'
#     # ligands_pdbqt_dir = 'data/ligands_pdbqt'
#     # docked_ligands_dir = 'data/docked_ligands_7uup_pocket'
    
#     ######PMDM finetuned 7l11 프로틴 데이터 도킹#######
#     # ligands_dir = 'outputs_pdb/PMDM_7l11_finetuned_gen_mol'
#     # ligands_pdbqt_dir = 'outputs_pdb/PMDM_7l11_finetuned_gen_mol/pdbqt'
#     # docked_ligands_dir = 'outputs_pdb/PMDM_7l11_finetuned_gen_mol/eval'
#     ######PMDM normal 7l11 프로틴 데이터 도킹#######
#     # ligands_dir = 'outputs_pdb/PMDM_7l11_normal'
#     # ligands_pdbqt_dir = 'outputs_pdb/PMDM_7l11_normal/pdbqt'
#     # docked_ligands_dir = 'outputs_pdb/PMDM_7l11_normal/eval'
#     ######Targetdiff 7uup normal 데이터 도킹#######
#     # ligands_dir = 'outputs_pdb/Pretrain_only_100'
#     # ligands_pdbqt_dir = 'outputs_pdb/Pretrain_only_100/pdbqt'
#     # docked_ligands_dir = 'outputs_pdb/Pretrain_only_100/7uup_whole_eval'
#     ######Targetdiff 7uup finetune_50000 데이터 도킹#######
#     # ligands_dir = 'outputs_pdb/finetune_7uup_pocket_100_iter50000'
#     # ligands_pdbqt_dir = 'outputs_pdb/finetune_7uup_pocket_100_iter50000/pdbqt'
#     # docked_ligands_dir = 'outputs_pdb/finetune_7uup_pocket_100_iter50000/7uup_pocket_eval'
#     ######Targetdiff 7l11 finetune_100 도킹#######
#     # ligands_dir = 'outputs_pdb/7l11_finetune_10000_100'
#     # ligands_pdbqt_dir = 'outputs_pdb/7l11_finetune_10000_100/pdbqt'
#     # docked_ligands_dir = 'outputs_pdb/7l11_finetune_10000_100/7l11_eval'
#     ######Targetdiff 7l11 normal_100 도킹#######
#     # ligands_dir = 'outputs_pdb/7l11_normal_100'
#     # ligands_pdbqt_dir = 'outputs_pdb/7l11_normal_100/pdbqt'
#     # docked_ligands_dir = 'outputs_pdb/7l11_normal_100/7l11_eval'
#     # ligands_dir = 'data/7l11_compound5_ligand'
#     # ligands_pdbqt_dir = 'data/7l11_compound5_ligand/pdbqt'
#     # docked_ligands_dir = 'data/7l11_compound5_ligand/result'
#     ligands_dir = 'data/7uup_nirmatrelvir_ligand'
#     ligands_pdbqt_dir = 'data/7uup_nirmatrelvir_ligand/pdbqt'
#     docked_ligands_dir = 'data/7uup_nirmatrelvir_ligand/result'

#     # ligands_pdbqt 디렉토리가 없으면 생성
#     if not os.path.exists(ligands_pdbqt_dir):
#         os.makedirs(ligands_pdbqt_dir)
#     # docked_ligands 디렉토리가 없으면 생성
#     if not os.path.exists(docked_ligands_dir):
#         os.makedirs(docked_ligands_dir)

#     # ligands_pdb 디렉토리 내의 PDB 파일 순차 처리 (파일 이름 순으로 정렬)
#     ligand_files = sorted(os.listdir(ligands_dir))
    
#     for ligand_file in ligand_files:
#         if ligand_file.endswith('.sdf'):
#             try:
#                 ligand_path = os.path.join(ligands_dir, ligand_file)
#                 ligand_pdbqt = os.path.join(ligands_pdbqt_dir, ligand_file.replace('.sdf', '.pdbqt'))
#                 # ligand_pdbqt = ligand_file.replace('.pdb', '.pdbqt')

#                 # 리간드 준비
#                 lig = PrepLig(ligand_path, 'sdf')
#                 lig.get_pdbqt(ligand_pdbqt)

#                 # 도킹 설정 및 실행
#                 dock = VinaDock(ligand_pdbqt, protein_pdbqt)

#                 # 도킹 박스 설정 (단백질 전체 사용)
#                 # dock.get_box(ref=None, buffer=5)  # 버퍼 크기 (Å)
#                 dock.get_box(center=docking_center, box_size=docking_box_size)

#                 # 도킹 수행
#                 score, pose = dock.dock(score_func='vina', seed=0, mode='dock', exhaustiveness=8, save_pose=True)

#                 # 도킹 결과 저장
#                 docked_ligand_pdbqt = f'{ligand_file.replace(".pdb", "")}_score_{score:.2f}.pdbqt'
#                 docked_ligand_path = os.path.join(docked_ligands_dir, docked_ligand_pdbqt)

#                 with open(docked_ligand_path, 'w') as f:
#                     f.write(pose)

#                 print(f'{ligand_file} 도킹 완료: 스코어 = {score} kcal/mol')
#                 print(f'도킹 결과가 {docked_ligand_path} 파일에 저장되었습니다.')

#             except Exception as e:
#                 print(f'{ligand_file} 처리 중 오류 발생: {str(e)}. 파일을 건너뜁니다.')



# if __name__ == '__main__':
#     main()
