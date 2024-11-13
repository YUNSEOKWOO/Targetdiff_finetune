# import os
# import pickle
# import lmdb
# from torch.utils.data import Dataset
# from tqdm import tqdm

# class PocketLigandPairDataset_finetune(Dataset):
#     def __init__(self, raw_path, transform=None):
#         super().__init__()
#         self.processed_path = raw_path.rstrip('/')
#         self.transform = transform
#         self.db = None
#         self.keys = None

#         if not os.path.exists(self.processed_path):
#             raise FileNotFoundError(f'{self.processed_path} does not exist.')

#         self._connect_db()

#     def _connect_db(self):
#         """
#         Establish read-only database connection
#         """
#         assert self.db is None, 'A connection has already been opened.'
#         self.db = lmdb.open(
#             self.processed_path,
#             map_size=10*(1024*1024*1024),   # 10GB
#             create=False,
#             subdir=False,
#             readonly=True,
#             lock=False,
#             readahead=False,
#             meminit=False,
#         )
#         with self.db.begin() as txn:
#             self.keys = list(txn.cursor().iternext(values=False))

#     def _close_db(self):
#         if self.db is not None:
#             self.db.close()
#             self.db = None
#             self.keys = None

#     def __len__(self):
#         return len(self.keys)

#     def __getitem__(self, idx):
#         if self.db is None:
#             self._connect_db()
#         data = self.get_ori_data(idx)
#         if self.transform is not None:
#             data = self.transform(data)
#         return data

#     def get_ori_data(self, idx):
#         key = self.keys[idx]
#         data = pickle.loads(self.db.begin().get(key))
#         # print(f"Data for key {key}: {data}")  # 데이터 구조 출력
#         return data
    
import os
import pickle
import lmdb
from torch.utils.data import Dataset
from tqdm import tqdm

from .pl_data import ProteinLigandData

class PocketLigandPairDataset_finetune(Dataset):
    def __init__(self, raw_path, transform=None):
        super().__init__()
        self.processed_path = raw_path.rstrip('/')
        self.transform = transform
        self.db = None
        self.keys = None

        if not os.path.exists(self.processed_path):
            raise FileNotFoundError(f'{self.processed_path} does not exist.')

        self._connect_db()

    def _connect_db(self):
        """
        Establish read-only database connection
        """
        assert self.db is None, 'A connection has already been opened.'
        self.db = lmdb.open(
            self.processed_path,
            map_size=10*(1024*1024*1024),   # 10GB
            create=False,
            subdir=False,
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.db.begin() as txn:
            self.keys = list(txn.cursor().iternext(values=False))

    def _close_db(self):
        if self.db is not None:
            self.db.close()
            self.db = None
            self.keys = None

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        if self.db is None:
            self._connect_db()
        data = self.get_ori_data(idx)
        if self.transform is not None:
            data = self.transform(data)
        return data

    def get_ori_data(self, idx):
        key = self.keys[idx]
        data_dict = pickle.loads(self.db.begin().get(key))
        data = ProteinLigandData(**data_dict)  # 데이터 객체로 변환
        data.id = idx  # 필요 시 추가 속성 설정
        assert data.protein_pos.size(0) > 0  # 데이터 검증
        return data
