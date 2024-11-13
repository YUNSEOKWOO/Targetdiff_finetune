import os
import glob
from rdkit import Chem
from rdkit.Chem import AllChem, Draw
import math
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.pyplot as plt

def visualize_sdf_to_pdf(sdf_directory, output_pdf=None, mols_per_page=25):
    """
    지정된 디렉토리 내의 모든 SDF 파일을 읽어 2D로 시각화하고 PDF로 저장합니다.

    :param sdf_directory: SDF 파일들이 있는 디렉토리 경로
    :param output_pdf: 출력할 PDF 파일 이름 (기본값: sdf_directory 이름 기반)
    :param mols_per_page: 한 페이지당 표시할 분자 수
    """
    # SDF 파일 목록 가져오기
    sdf_files = glob.glob(os.path.join(sdf_directory, '*.sdf'))
    if not sdf_files:
        print(f"디렉토리 '{sdf_directory}'에 SDF 파일이 존재하지 않습니다.")
        return

    molecules = []
    for sdf_file in sdf_files:
        print(f"파일 읽는 중: {sdf_file}")
        suppl = Chem.SDMolSupplier(sdf_file)
        if suppl is None:
            print(f"파일을 읽는 데 실패했습니다: {sdf_file}")
            continue
        for mol in suppl:
            if mol is not None:
                # 2D 좌표 계산
                AllChem.Compute2DCoords(mol)
                molecules.append(mol)
            else:
                print(f"{sdf_file}에서 유효하지 않은 분자를 발견했습니다.")

    if not molecules:
        print("유효한 분자를 찾을 수 없습니다.")
        return

    print(f"총 {len(molecules)}개의 분자를 시각화합니다.")

    # PDF 파일 경로 및 이름 설정
    if output_pdf is None:
        # sdf_directory의 이름을 가져와서 PDF 파일 이름으로 사용
        directory_name = os.path.basename(os.path.normpath(sdf_directory))
        output_pdf = os.path.join(sdf_directory, f"{directory_name}.pdf")

    # PDF 파일 열기
    with PdfPages(output_pdf) as pdf:
        # 전체 페이지 수 계산
        total_pages = math.ceil(len(molecules) / mols_per_page)
        for page in range(total_pages):
            start = page * mols_per_page
            end = start + mols_per_page
            mols = molecules[start:end]

            # 그리드 이미지 생성
            img = Draw.MolsToGridImage(
                mols,
                molsPerRow=5,           # 한 행에 표시할 분자 수
                subImgSize=(200, 200),  # 각 분자의 이미지 크기
                useSVG=False
            )

            # matplotlib을 사용하여 이미지 추가
            plt.figure(figsize=(11.69, 8.27))  # A4 용지 크기 (인치 단위)
            plt.axis('off')
            plt.imshow(img)
            plt.tight_layout()
            pdf.savefig()
            plt.close()

    print(f"모든 분자가 '{output_pdf}' 파일에 저장되었습니다.")

if __name__ == '__main__':
    sdf_directory = 'outputs_pdb/7l11_normal_100'  # SDF 파일들이 있는 디렉토리
    visualize_sdf_to_pdf(sdf_directory)


# import os
# import glob
# from rdkit import Chem
# from rdkit.Chem import AllChem, Draw
# import math
# from matplotlib.backends.backend_pdf import PdfPages
# import matplotlib.pyplot as plt

# def visualize_sdf_to_pdf(sdf_directory, output_pdf, mols_per_page=25):
#     """
#     지정된 디렉토리 내의 모든 SDF 파일을 읽어 2D로 시각화하고 PDF로 저장합니다.

#     :param sdf_directory: SDF 파일들이 있는 디렉토리 경로
#     :param output_pdf: 출력할 PDF 파일 이름
#     :param mols_per_page: 한 페이지당 표시할 분자 수
#     """
#     # SDF 파일 목록 가져오기
#     sdf_files = glob.glob(os.path.join(sdf_directory, '*.sdf'))
#     if not sdf_files:
#         print(f"디렉토리 '{sdf_directory}'에 SDF 파일이 존재하지 않습니다.")
#         return

#     molecules = []
#     for sdf_file in sdf_files:
#         print(f"파일 읽는 중: {sdf_file}")
#         suppl = Chem.SDMolSupplier(sdf_file)
#         if suppl is None:
#             print(f"파일을 읽는 데 실패했습니다: {sdf_file}")
#             continue
#         for mol in suppl:
#             if mol is not None:
#                 # 2D 좌표 계산
#                 AllChem.Compute2DCoords(mol)
#                 molecules.append(mol)
#             else:
#                 print(f"{sdf_file}에서 유효하지 않은 분자를 발견했습니다.")

#     if not molecules:
#         print("유효한 분자를 찾을 수 없습니다.")
#         return

#     print(f"총 {len(molecules)}개의 분자를 시각화합니다.")

#     # PDF 파일 열기
#     with PdfPages(output_pdf) as pdf:
#         # 전체 페이지 수 계산
#         total_pages = math.ceil(len(molecules) / mols_per_page)
#         for page in range(total_pages):
#             start = page * mols_per_page
#             end = start + mols_per_page
#             mols = molecules[start:end]

#             # 그리드 이미지 생성
#             img = Draw.MolsToGridImage(
#                 mols,
#                 molsPerRow=5,           # 한 행에 표시할 분자 수
#                 subImgSize=(200, 200),  # 각 분자의 이미지 크기
#                 useSVG=False
#             )

#             # matplotlib을 사용하여 이미지 추가
#             plt.figure(figsize=(11.69, 8.27))  # A4 용지 크기 (인치 단위)
#             plt.axis('off')
#             plt.imshow(img)
#             plt.tight_layout()
#             pdf.savefig()
#             plt.close()

#     print(f"모든 분자가 '{output_pdf}' 파일에 저장되었습니다.")

# if __name__ == '__main__':
#     sdf_directory = 'outputs_pdb/Pretrain_only_100'      # SDF 파일들이 있는 디렉토리
#     output_pdf = 'all_molecules.pdf'       # 출력할 PDF 파일 이름
#     visualize_sdf_to_pdf(sdf_directory, output_pdf)
