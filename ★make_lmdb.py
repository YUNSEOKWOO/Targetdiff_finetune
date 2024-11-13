import os
import torch
import lmdb
import pickle
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np
from openbabel import openbabel
import logging
import pandas as pd

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_ligand_features_from_pdbqt(pdbqt_file):
    """
    OpenBabel과 RDKit을 사용하여 PDBQT 파일에서 리간드의 결합 정보 및 원자 특징을 추출합니다.

    Args:
        pdbqt_file (str): PDBQT 파일 경로.

    Returns:
        dict: 리간드의 결합 정보, 원자 특징, 하이브리드화 상태, 이웃 원자 리스트, 원자 번호, 좌표 등을 포함합니다.
    """
    # OpenBabel 초기화
    obConversion = openbabel.OBConversion()
    obConversion.SetInAndOutFormats("pdbqt", "mol")

    obMol = openbabel.OBMol()
    if not obConversion.ReadFile(obMol, pdbqt_file):
        raise ValueError(f"Failed to read molecule from {pdbqt_file}")

    # 결합 순위 재계산
    obMol.ConnectTheDots()
    obMol.PerceiveBondOrders()
    obMol.FindRingAtomsAndBonds()

    # 임시 .mol 파일 경로
    mol_output = pdbqt_file.replace('.pdbqt', '.mol')

    # .mol 파일로 저장
    if not obConversion.WriteFile(obMol, mol_output):
        raise ValueError(f"Failed to write molecule to {mol_output}")

    # RDKit으로 .mol 파일 읽기
    mol = Chem.MolFromMolFile(mol_output, removeHs=False)
    if mol is None:
        raise ValueError(f"Failed to read molecule from {mol_output}")

    # 결합 인덱스와 타입 추출
    bond_indices = []
    bond_types = []
    for bond in mol.GetBonds():
        bond_indices.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        bond_types.append(int(bond.GetBondTypeAsDouble()))

    bond_indices = torch.tensor(bond_indices, dtype=torch.long).t()
    bond_types = torch.tensor(bond_types, dtype=torch.long)

    # 원자 번호와 좌표 추출
    ligand_element = []
    ligand_pos = []
    conf = mol.GetConformer()
    positions = conf.GetPositions()
    for atom in mol.GetAtoms():
        ligand_element.append(atom.GetAtomicNum())
        ligand_pos.append(positions[atom.GetIdx()])

    ligand_element = torch.tensor(ligand_element, dtype=torch.long)
    ligand_pos = torch.tensor(ligand_pos, dtype=torch.float32)

    # 중심 좌표 계산 (필요 시 사용)
    center_of_mass = torch.tensor(np.mean(positions, axis=0), dtype=torch.float32)

    # 원자 특징 및 하이브리드화 상태 추출
    atom_features = []
    hybridizations = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),          # 원자 번호
            atom.GetFormalCharge(),       # 정전하
            atom.GetTotalNumHs(),         # 수소 원자 수
            int(atom.GetIsAromatic()),    # 아로마틱 여부
        ]
        atom_features.append(features)
        hybridizations.append(int(atom.GetHybridization()))

    atom_features = torch.tensor(atom_features, dtype=torch.float)
    hybridizations = torch.tensor(hybridizations, dtype=torch.long)

    # 인접 원자 리스트 추출
    nbh_list = []
    for atom in mol.GetAtoms():
        neighbors = [nbr.GetIdx() for nbr in atom.GetNeighbors()]
        nbh_list.append(neighbors)

    # 임시 .mol 파일 삭제 (선택 사항)
    try:
        os.remove(mol_output)
    except OSError as e:
        logger.warning(f"Could not remove temporary file {mol_output}: {e}")

    return {
        'ligand_bond_index': bond_indices,
        'ligand_bond_type': bond_types,
        'ligand_center_of_mass': center_of_mass,
        'ligand_atom_feature': atom_features,
        'ligand_hybridization': hybridizations,
        'ligand_nbh_list': nbh_list,
        'ligand_element': ligand_element,
        'ligand_pos': ligand_pos,
    }

def parse_pdb(pdb_file):
    """
    PDB 파일을 파싱하여 단백질의 원자 정보를 추출합니다.

    Args:
        pdb_file (str): PDB 파일 경로.

    Returns:
        dict: 단백질의 원자 요소, 이름, 위치, 백본 여부, 아미노산 타입 등을 포함합니다.
    """
    protein_element = []
    protein_atom_name = []
    protein_pos = []
    protein_is_backbone = []
    protein_atom_to_aa_type = []

    # Map residue names to amino acid types
    aa_mapping = {'SER': 1, 'GLY': 2, 'PHE': 3, 'ARG': 4, 'LYS': 5, 'MET': 6}
    # 필요한 모든 아미노산을 매핑에 추가해야 합니다.

    with open(pdb_file, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                atom_name = line[12:16].strip()
                residue_name = line[17:20].strip()
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                atom_type = line[76:78].strip()

                element_mapping = {'C': 6, 'N': 7, 'O': 8, 'S': 16}
                protein_element.append(element_mapping.get(atom_type, 0))

                protein_atom_name.append(atom_name)
                protein_pos.append([x, y, z])

                is_backbone = atom_name in {'N', 'CA', 'C', 'O'}
                protein_is_backbone.append(is_backbone)

                aa_type = aa_mapping.get(residue_name, 0)
                protein_atom_to_aa_type.append(aa_type)

    protein_element = torch.tensor(protein_element, dtype=torch.long)
    protein_pos = torch.tensor(protein_pos, dtype=torch.float32)
    protein_is_backbone = torch.tensor(protein_is_backbone, dtype=torch.bool)
    protein_atom_to_aa_type = torch.tensor(protein_atom_to_aa_type, dtype=torch.long)

    return {
        'protein_element': protein_element,
        'protein_atom_name': protein_atom_name,
        'protein_pos': protein_pos,
        'protein_is_backbone': protein_is_backbone,
        'protein_atom_to_aa_type': protein_atom_to_aa_type,
    }

def get_smiles_from_csv(csv_file, ligand_index):
    """
    CSV 파일에서 리간드 인덱스에 해당하는 SMILES 문자열을 가져옵니다.

    Args:
        csv_file (str): CSV 파일 경로.
        ligand_index (int): 리간드 인덱스.

    Returns:
        str or None: 해당 인덱스의 SMILES 문자열 또는 None.
    """
    df = pd.read_csv(csv_file)
    # 'index' 열이 있다고 가정하고, 해당 인덱스의 SMILES를 반환
    smiles_row = df[df['index'] == ligand_index]
    if not smiles_row.empty:
        return smiles_row['smiles'].values[0]
    else:
        return None

def create_lmdb(lmdb_path, protein_data, ligand_data_dict):
    """
    여러 리간드 데이터를 LMDB 파일에 저장합니다.

    Args:
        lmdb_path (str): LMDB 파일 경로.
        protein_data (dict): 단백질 데이터.
        ligand_data_dict (dict): 리간드 데이터의 사전, 키는 LMDB의 키가 됩니다.
    """
    map_size = 10 * (1024 ** 3)  # 10 GB
    db = lmdb.open(lmdb_path, map_size=map_size, subdir=False, readonly=False, meminit=False, map_async=True)

    with db.begin(write=True) as txn:
        for key, ligand_data in ligand_data_dict.items():
            data = {
                'ligand_element': ligand_data['ligand_element'],
                'ligand_pos': ligand_data['ligand_pos'],
                'ligand_bond_index': ligand_data['ligand_bond_index'],
                'ligand_bond_type': ligand_data['ligand_bond_type'],
                'ligand_center_of_mass': ligand_data['ligand_center_of_mass'],
                'ligand_atom_feature': ligand_data['ligand_atom_feature'],
                'ligand_hybridization': ligand_data['ligand_hybridization'],
                'ligand_nbh_list': ligand_data['ligand_nbh_list'],
                'ligand_smiles': ligand_data['ligand_smiles'],
                'protein_element': protein_data['protein_element'],
                'protein_atom_name': protein_data['protein_atom_name'],
                'protein_pos': protein_data['protein_pos'],
                'protein_is_backbone': protein_data['protein_is_backbone'],
                'protein_atom_to_aa_type': protein_data['protein_atom_to_aa_type'],
            }

            # Serialize data with pickle
            serialized_data = pickle.dumps(data)

            # Store data in LMDB with the ligand index as key
            txn.put(key.encode('utf-8'), serialized_data)

            logger.info(f"Stored ligand with key: {key}")

    db.sync()
    db.close()
    logger.info(f"LMDB file created at: {lmdb_path}")

if __name__ == '__main__':
    # 파일 경로 설정
    ligand_dir = 'data/docked_ligands_7uup_pocket'
    protein_pdb_file = 'data/target_pdb/7uup_B_4WI_pocket10.pdb'
    lmdb_path = 'data/output_ligand_protein_7uup_B_4WI_pocket10.lmdb'
    csv_file = 'data/ligands_pdb/positive_ligands.csv'

    # 단백질 데이터 파싱
    logger.info("Parsing protein PDB file...")
    protein_data = parse_pdb(protein_pdb_file)
    logger.info("Protein data parsed successfully.")

    # LMDB에 저장할 리간드 데이터 사전 초기화
    ligand_data_dict = {}

    # 리간드 PDBQT 파일들을 순회하며 데이터 추출
    logger.info("Starting to process ligand PDBQT files...")
    for ligand_file in os.listdir(ligand_dir):
        if ligand_file.endswith('.pdbqt'):
            ligand_pdbqt_path = os.path.join(ligand_dir, ligand_file)

            try:
                # 파일 이름에서 리간드 인덱스 추출 (예: '1_ligand_score_-8.52.pdbqt' -> 1)
                ligand_index_str = os.path.basename(ligand_pdbqt_path).split('_')[0]
                ligand_index = int(ligand_index_str)

                # CSV에서 해당 인덱스의 SMILES 가져오기
                ligand_smiles = get_smiles_from_csv(csv_file, ligand_index)
                if ligand_smiles is None:
                    logger.warning(f"SMILES not found for index: {ligand_index}. Skipping this ligand.")
                    continue

                # PDBQT에서 리간드 특징 추출
                ligand_features = extract_ligand_features_from_pdbqt(ligand_pdbqt_path)

                # 리간드 데이터를 사전에 추가
                ligand_data_dict[str(ligand_index)] = {
                    'ligand_element': ligand_features['ligand_element'],
                    'ligand_pos': ligand_features['ligand_pos'],
                    'ligand_bond_index': ligand_features['ligand_bond_index'],
                    'ligand_bond_type': ligand_features['ligand_bond_type'],
                    'ligand_center_of_mass': ligand_features['ligand_center_of_mass'],
                    'ligand_atom_feature': ligand_features['ligand_atom_feature'],
                    'ligand_hybridization': ligand_features['ligand_hybridization'],
                    'ligand_nbh_list': ligand_features['ligand_nbh_list'],
                    'ligand_smiles': ligand_smiles,
                }

                logger.info(f"Processed ligand index: {ligand_index}")

            except Exception as e:
                logger.error(f"Error processing {ligand_file}: {e}")
                continue

    # LMDB에 모든 리간드 데이터 저장
    logger.info("Storing all ligand data into LMDB...")
    create_lmdb(lmdb_path, protein_data, ligand_data_dict)
    logger.info("All data stored successfully.")