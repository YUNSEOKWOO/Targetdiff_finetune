import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# CSV 파일 경로 설정
csv_file_path = 'Final_ligands12_MD.csv'
df = pd.read_csv(csv_file_path)

# CSV 파일이 있는 디렉토리로부터 PDF 파일 경로 생성
csv_dir = os.path.dirname(csv_file_path)
pdf_filename = os.path.join(csv_dir, 'molecule_visualization.pdf')

# PDF 파일 생성
pdf_pages = PdfPages(pdf_filename)

# 시각화 시작 (한 페이지에 한 개의 분자)
for i in range(len(df)):
    fig, ax = plt.subplots(figsize=(8, 8))  # 페이지 크기 및 해상도를 크게 설정

    # SMILES에서 Molecule 객체 생성
    row = df.iloc[i]
    model = row['model']
    smiles = row['SMILES']
    mol = Chem.MolFromSmiles(smiles)

    if mol is not None:
        # Molecule 2D 이미지 생성
        img = Draw.MolToImage(mol, size=(400, 400))  # 해상도 향상
        ax.imshow(img)
        ax.axis('off')

        # 레이블 추가 (그림 하단에 배치)
        label = f"{model}\n{smiles}\ngen_id: {row['gen_id']}\nQED: {row['QED']:.3f}  SAS: {row['SAS']:.3f}  Vina Score: {row['vina_score']}  pIC50: {row['SVR_predicted_pIC50']}"
        ax.text(0.5, -0.1, label, fontsize=12, ha='center', transform=ax.transAxes, wrap=True)

    else:
        ax.axis('off')  # 분자가 생성되지 않으면 빈 페이지 처리

    # 페이지를 PDF로 저장 (간격을 조정하기 위해 tight_layout 사용)
    plt.tight_layout(pad=2.0)  # pad 값을 줄여서 그림과 레이블 간격 조정
    pdf_pages.savefig(fig)
    plt.close(fig)

# PDF 저장 완료
pdf_pages.close()
print(f"PDF 파일이 '{pdf_filename}' 이름으로 저장되었습니다.")





#######기존 코드(레이블링 추가해서 PDF로 한꺼번에 저장)
# import os
# import pandas as pd
# from rdkit import Chem
# from rdkit.Chem import Draw
# import matplotlib.pyplot as plt
# from matplotlib.backends.backend_pdf import PdfPages

# # CSV 파일 경로 설정
# csv_file_path = 'outputs_pdb/finetune_7uup_pocket_100_iter50000/eval_results/per_molecule_metrics_-1.csv'
# df = pd.read_csv(csv_file_path)

# # CSV 파일이 있는 디렉토리로부터 PDF 파일 경로 생성
# csv_dir = os.path.dirname(csv_file_path)
# pdf_filename = os.path.join(csv_dir, 'molecule_visualization.pdf')

# # PDF 파일 생성
# pdf_pages = PdfPages(pdf_filename)

# # 페이지 당 시각화할 분자의 수
# molecules_per_page = 6
# # 한 페이지에 그릴 행과 열의 개수
# rows, cols = 3, 3

# # 시각화 시작
# for i in range(0, len(df), molecules_per_page):
#     fig, axes = plt.subplots(rows, cols, figsize=(12, 8))
#     axes = axes.flatten()

#     for j, ax in enumerate(axes):
#         idx = i + j
#         if idx < len(df):
#             # SMILES에서 Molecule 객체 생성
#             row = df.iloc[idx]
#             smiles = row['smiles']
#             mol = Chem.MolFromSmiles(smiles)

#             if mol is not None:
#                 # Molecule 2D 이미지 생성
#                 img = Draw.MolToImage(mol, size=(200, 200))
#                 ax.imshow(img)
#                 ax.axis('off')

#                 # 레이블 추가 (그림 하단에 배치)
#                 label = f"Index: {idx}\n{smiles}\nQED: {row['qed']:.3f}  SA: {row['sa']:.3f}  Vina Score: {row['vina_score']}"
#                 ax.text(0.5, -0.1, label, fontsize=10, ha='center', transform=ax.transAxes, wrap=True)  # 레이블 추가 및 간격 조정

#         else:
#             ax.axis('off')  # 빈 칸은 비워둠

#     # 페이지를 PDF로 저장 (간격을 조정하기 위해 tight_layout 사용)
#     plt.tight_layout(pad=2.0)  # pad 값을 줄여서 그림과 레이블 간격 조정
#     pdf_pages.savefig(fig)
#     plt.close(fig)

# # PDF 저장 완료
# pdf_pages.close()
# print(f"PDF 파일이 '{pdf_filename}' 이름으로 저장되었습니다.")
