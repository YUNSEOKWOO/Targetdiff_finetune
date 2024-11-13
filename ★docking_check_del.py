import os
import pandas as pd

# 디렉토리 경로 설정
directory = 'data/docked_ligands_7l11_autodock'

# 파일명에서 index 추출
index_numbers = []

for filename in os.listdir(directory):
    if filename.endswith(".pdbqt"):
        index_number = filename.split('_')[0]
        index_numbers.append(index_number)

# DataFrame으로 변환
df = pd.DataFrame(index_numbers, columns=["index_number"])

# CSV 파일로 저장
csv_file_path = 'index_numbers.csv'
df.to_csv(csv_file_path, index=False)

# 결과 확인
df.head()
