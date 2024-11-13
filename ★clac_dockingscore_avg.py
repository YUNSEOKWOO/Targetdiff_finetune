import os
import re

def calculate_average_score(directory):
    # pdbqt 파일을 포함하는 디렉토리
    scores = []
    
    # 디렉토리 내 파일들 순회
    for filename in os.listdir(directory):
        # pdbqt 파일만 처리
        if filename.endswith(".pdbqt"):
            # 파일 이름에서 스코어 추출 (정규 표현식 사용)
            match = re.search(r"_score_(-?\d+\.\d+)\.pdbqt", filename)
            if match:
                # 스코어를 실수로 변환하여 리스트에 추가
                score = float(match.group(1))
                scores.append(score)
    
    # 평균 계산
    if scores:
        avg_score = sum(scores) / len(scores)
        return avg_score
    else:
        return None

# 디렉토리 경로 설정
# directory = "outputs_pdb/PMDM_7l11_normal/eval"
# directory = "outputs_pdb/PMDM_7l11_finetuned_gen_mol/eval"
# directory = "outputs_pdb/Pretrain_only_100/7uup_pocket_eval"
# directory = "outputs_pdb/Pretrain_only_100/7uup_whole_eval"
# directory = "outputs_pdb/finetune_7uup_pocket_100_iter50000/7uup_pocket_eval"
directory = "outputs_pdb/7l11_normal_100/7l11_eval"

# 평균 스코어 계산
average_score = calculate_average_score(directory)

# 결과 출력
if average_score is not None:
    print(f"{directory}   평균 스코어: {average_score:.2f}")
else:
    print("pdbqt 파일에서 스코어를 찾을 수 없습니다.")
