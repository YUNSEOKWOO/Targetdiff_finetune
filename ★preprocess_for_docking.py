from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolTransforms
import pubchempy as pcp
import pandas as pd
import openbabel
import os

# def addH(self, polaronly=False, correctforph=True, PH=7): 
#     self.ob_mol.OBMol.AddHydrogens(polaronly, correctforph, PH)
#     obutils.writeMolecule(self.ob_mol.OBMol, 'tmp_h.sdf')
    

def smiles_to_pdb(smiles, output_file):
    """
    PubChem에서 SMILES로부터 3D 구조를 가져오고, 실패 시 RDKit을 사용해 3D 구조를 생성하여 PDB 파일로 저장하는 함수
    """
    try:
        # PubChem에서 SMILES로 화합물 검색
        compound = pcp.get_compounds(smiles, 'smiles', record_type='3d')

        if compound and compound[0].atoms:
            atoms = compound[0].atoms  # 화합물의 원자 목록 가져오기
            # OpenBabel로 PDB 파일 저장
            obConversion = openbabel.OBConversion()
            obConversion.SetOutFormat("pdb")
            mol = openbabel.OBMol()

            for atom in atoms:
                ob_atom = mol.NewAtom()
                ob_atom.SetAtomicNum(atom.number)
                if atom.x and atom.y and atom.z:
                    ob_atom.SetVector(atom.x, atom.y, atom.z)  # 좌표 설정
                else:
                    print(f"3D 좌표를 찾을 수 없습니다: {smiles}")
                    return
                
            obConversion.WriteFile(mol, output_file)
            print(f"{output_file} 저장 완료.")
        else:
            # PubChem에서 찾지 못했을 때 RDKit으로 3D 구조 생성
            print(f"PubChem에서 화합물을 찾을 수 없습니다. RDKit을 사용하여 3D 구조 생성.")
            rdkit_generate_3d(smiles, output_file)

    except Exception as e:
        print(f"오류 발생: {e}")


def rdkit_generate_3d(smiles, output_file):
    """
    RDKit을 사용하여 SMILES로부터 3D 구조를 생성하고 PDB 파일로 저장하는 함수
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            print(f"SMILES 문자열이 유효하지 않습니다: {smiles}")
            return

        # 수소 원자 추가
        mol = Chem.AddHs(mol)

        # 3D 좌표 생성
        AllChem.EmbedMolecule(mol, AllChem.ETKDG())

        # 좌표 최적화
        AllChem.UFFOptimizeMolecule(mol)

        # PDB 형식으로 저장
        with open(output_file, 'w') as f:
            f.write(Chem.MolToPDBBlock(mol))
        print(f"{output_file} 저장 완료.")
    
    except Exception as e:
        print(f"RDKit으로 3D 구조 생성 중 오류 발생: {e}")


def process_csv(input_csv, smiles_column):
    """
    CSV 파일에서 SMILES 열을 읽고, 각 SMILES에 대해 PubChem에서 3D 구조를 가져와 PDB 파일로 저장
    """
    # CSV 파일 읽기
    data = pd.read_csv(input_csv)

    if smiles_column not in data.columns:
        raise ValueError(f"CSV 파일에 '{smiles_column}' 열이 존재하지 않습니다.")
    
    # input_csv에서 파일명만 추출
    csv_filename = os.path.basename(input_csv).split(".")[0]  # 파일명에서 확장자를 제외한 이름 추출

    # CSV 파일명으로 된 디렉토리 생성
    output_dir = f"outputs_pdb/drugGPT_VAE/{csv_filename}_pdb"
    os.makedirs(output_dir, exist_ok=True)  # 이미 존재해도 에러 발생하지 않음

    # 각 SMILES에 대해 처리
    for index, row in data.iterrows():
        smiles = row[smiles_column]

        # PDB 파일명 설정: CSV 파일명 기반 디렉토리 내에 저장
        output_file = os.path.join(output_dir, f"{index + 1}_ligand.pdb")
        
        # SMILES를 PDB로 변환하는 함수 호출 (구현되지 않은 부분)
        smiles_to_pdb(smiles, output_file)

# CSV 파일 처리
# process_csv("data/ligands_pdb/positive_ligands.csv", "smiles")
process_csv("outputs_pdb/drugGPT_VAE/druggpt_finetuned_100.csv", "SMILES")
process_csv("outputs_pdb/drugGPT_VAE/druggpt_normal_100.csv", "SMILES")
process_csv("outputs_pdb/drugGPT_VAE/vae_finetuned_100.csv", "SMILES")