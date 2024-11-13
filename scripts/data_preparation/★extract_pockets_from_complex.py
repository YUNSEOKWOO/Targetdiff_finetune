# import os
# import argparse
# import shutil

# from tqdm.auto import tqdm

# from utils.data import PDBProtein, parse_sdf_file  # 필요한 유틸리티 함수 임포트

# # RDKit을 이용하여 SDF를 PDB로 변환 (좌표 변경 없이)
# from rdkit import Chem

# def load_pdb(pdb_path):
#     with open(pdb_path, 'r') as f:
#         return f.read()

# def load_sdf(sdf_path):
#     return parse_sdf_file(sdf_path)

# def extract_pocket(pdb_block, ligand, radius):
#     protein = PDBProtein(pdb_block)
#     pocket_pdb = protein.residues_to_pdb_block(
#         protein.query_residues_ligand(ligand, radius)
#     )
#     return pocket_pdb

# def convert_sdf_to_pdb_preserve_coordinates(sdf_path, pdb_path):
#     """RDKit을 사용하여 SDF 파일을 PDB 파일로 변환 (좌표 변경 없이)"""
#     mol = Chem.MolFromMolFile(sdf_path, removeHs=False)
#     if mol is None:
#         raise ValueError(f"RDKit이 SDF 파일 '{sdf_path}'를 읽을 수 없습니다.")
    
#     # 수소 원자 추가 (필요 시)
#     mol = Chem.AddHs(mol)
    
#     # 좌표 변경 없이 PDB 파일로 저장
#     Chem.MolToPDBFile(mol, pdb_path)

# def main(args):
#     # PDB 파일과 SDF 파일 경로 확인
#     if not os.path.isfile(args.pdb):
#         print(f"Error: PDB file '{args.pdb}' does not exist.")
#         return
#     if not os.path.isfile(args.sdf):
#         print(f"Error: SDF file '{args.sdf}' does not exist.")
#         return

#     # PDB 및 SDF 파일 로드
#     pdb_block = load_pdb(args.pdb)
#     ligand = load_sdf(args.sdf)

#     # 포켓 추출
#     pocket_pdb = extract_pocket(pdb_block, ligand, args.radius)

#     # 출력 파일 이름 생성
#     ligand_fn = os.path.basename(args.sdf)
#     pocket_fn = os.path.splitext(ligand_fn)[0] + f'_pocket{args.radius}.pdb'
#     complex_fn = os.path.splitext(ligand_fn)[0] + f'_complex.pdb'

#     # 목적지 디렉토리 생성
#     os.makedirs(args.dest, exist_ok=True)

#     # 리간드 SDF 파일을 목적지로 복사
#     ligand_dest = os.path.join(args.dest, ligand_fn)
#     shutil.copyfile(src=args.sdf, dst=ligand_dest)

#     # 포켓 PDB 파일 저장
#     pocket_dest = os.path.join(args.dest, pocket_fn)
#     with open(pocket_dest, 'w') as f:
#         f.write(pocket_pdb)

#     print(f"포켓 추출 완료: {pocket_dest}")
#     print(f"리간드 파일 복사 완료: {ligand_dest}")

#     # 리간드 SDF를 PDB로 변환 (좌표 변경 없이)
#     ligand_pdb = os.path.splitext(ligand_fn)[0] + '.pdb'
#     ligand_pdb_path = os.path.join(args.dest, ligand_pdb)
#     try:
#         convert_sdf_to_pdb_preserve_coordinates(ligand_dest, ligand_pdb_path)
#         print(f"리간드 PDB 변환 완료: {ligand_pdb_path}")
#     except ValueError as e:
#         print(f"리간드 PDB 변환 실패: {e}")
#         return

#     # 포켓 PDB와 리간드 PDB를 결합하여 복합체 PDB 생성
#     complex_pdb_path = os.path.join(args.dest, complex_fn)
#     with open(complex_pdb_path, 'w') as complex_file:
#         # 포켓 PDB 내용 쓰기
#         with open(pocket_dest, 'r') as pocket_file:
#             shutil.copyfileobj(pocket_file, complex_file)
        
#         # 리간드 PDB 내용 쓰기
#         with open(ligand_pdb_path, 'r') as ligand_file:
#             for line in ligand_file:
#                 if line.startswith("END"):
#                     continue  # END 라인은 제외
#                 complex_file.write(line)

#     print(f"복합체 PDB 생성 완료: {complex_pdb_path}")

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser(description="단일 단백질-리간드 쌍의 포켓 추출 및 복합체 PDB 생성 스크립트")
#     parser.add_argument('--pdb', type=str, default='data/mpro_complex/7uup.pdb', help="단백질 PDB 파일 경로")
#     parser.add_argument('--sdf', type=str, default='data/mpro_complex/7uup_B_4WI.sdf', help="리간드 SDF 파일 경로")
#     parser.add_argument('--dest', type=str, default='data/mpro_complex/pocket', help="출력 파일이 저장될 목적지 디렉토리")
#     parser.add_argument('--radius', type=int, default=10, help="리간드 주변 포켓 탐색 반경 (Å)")
#     args = parser.parse_args()

#     main(args)








import os
import argparse
import shutil

from tqdm.auto import tqdm

from utils.data import PDBProtein, parse_sdf_file  # 필요한 유틸리티 함수 임포트

def load_pdb(pdb_path):
    with open(pdb_path, 'r') as f:
        return f.read()

def load_sdf(sdf_path):
    return parse_sdf_file(sdf_path)

def extract_pocket(pdb_block, ligand, radius):
    protein = PDBProtein(pdb_block)
    pocket_pdb = protein.residues_to_pdb_block(
        protein.query_residues_ligand(ligand, radius)
    )
    return pocket_pdb

def main(args):
    # PDB 파일과 SDF 파일 경로 확인
    if not os.path.isfile(args.pdb):
        print(f"Error: PDB file '{args.pdb}' does not exist.")
        return
    if not os.path.isfile(args.sdf):
        print(f"Error: SDF file '{args.sdf}' does not exist.")
        return

    # PDB 및 SDF 파일 로드
    pdb_block = load_pdb(args.pdb)
    ligand = load_sdf(args.sdf)

    # 포켓 추출
    pocket_pdb = extract_pocket(pdb_block, ligand, args.radius)

    # 출력 파일 이름 생성
    ligand_fn = os.path.basename(args.sdf)
    pocket_fn = os.path.splitext(ligand_fn)[0] + f'_pocket{args.radius}.pdb'

    # 목적지 디렉토리 생성
    os.makedirs(args.dest, exist_ok=True)

    # 리간드 SDF 파일을 목적지로 복사
    ligand_dest = os.path.join(args.dest, ligand_fn)
    shutil.copyfile(src=args.sdf, dst=ligand_dest)

    # 포켓 PDB 파일 저장
    pocket_dest = os.path.join(args.dest, pocket_fn)
    with open(pocket_dest, 'w') as f:
        f.write(pocket_pdb)

    print(f"포켓 추출 완료: {pocket_dest}")
    print(f"리간드 파일 복사 완료: {ligand_dest}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="단일 단백질-리간드 쌍의 포켓 추출 및 복합체 PDB 생성 스크립트")
    # parser.add_argument('--pdb', type=str, default='data/mpro_complex/7uup.pdb', help="단백질 PDB 파일 경로")
    # parser.add_argument('--sdf', type=str, default='data/mpro_complex/7uup_B_4WI.sdf', help="리간드 SDF 파일 경로")
    # parser.add_argument('--dest', type=str, default='data/mpro_complex/pocket', help="출력 파일이 저장될 목적지 디렉토리")
    parser.add_argument('--pdb', type=str, default='data/target_pdb/7l11_A_protein.pdb', help="단백질 PDB 파일 경로")
    parser.add_argument('--sdf', type=str, default='data/7l11_compound5_ligand/7l11_C_XF1.sdf', help="리간드 SDF 파일 경로")
    parser.add_argument('--dest', type=str, default='data/mpro_complex/pocket', help="출력 파일이 저장될 목적지 디렉토리")
    parser.add_argument('--radius', type=int, default=10, help="리간드 주변 포켓 탐색 반경 (Å)")
    args = parser.parse_args()

    main(args)