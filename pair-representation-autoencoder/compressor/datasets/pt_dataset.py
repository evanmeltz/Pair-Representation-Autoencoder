from typing import Callable, List, Optional, Sequence, Union
import os.path as osp

import pandas as pd
from mmengine.dataset.base_dataset import BaseDataset as BaseDataset_
from mmengine.registry import DATASETS


@DATASETS.register_module()
class CsvPtDataset(BaseDataset_):
    """Dataset backed by a CSV index of NumPy pair-embedding files.

    The CSV must contain a ``path`` column. Each path is joined with
    ``path_prefix`` and loaded by the configured pipeline. The referenced NPZ
    file must contain a ``pair_embeddings`` array with shape ``(L, L, C)``.
    """

    def __init__(
        self,
        ann_file: str = "",
        data_root: str = "",
        data_prefix: dict = dict(),
        path_prefix: str = "",
        filter_cfg: Optional[dict] = None,
        indices: Optional[Union[int, Sequence[int]]] = None,
        pipeline: List[Union[dict, Callable]] = [],
        test_mode: bool = False,
        lazy_init: bool = False,
        max_refetch: int = 1000,
        serialize_data: bool = False,
    ):
        self.path_prefix = path_prefix
        
        super().__init__(
            ann_file=ann_file,
            metainfo=None,
            data_root=data_root,
            data_prefix=data_prefix,
            filter_cfg=filter_cfg,
            indices=indices,
            serialize_data=serialize_data,
            pipeline=pipeline,
            test_mode=test_mode,
            lazy_init=lazy_init,
            max_refetch=max_refetch,
        )

    def load_data_list(self) -> List[dict]:
        """Load sample paths from the CSV ``path`` column."""
        df = pd.read_csv(self.ann_file)
        
        data_list = []
        for _, row in df.iterrows():
            rel_path = row["path"]
            sample_path = osp.join(self.path_prefix, rel_path)
            data_info = {
                "sample_path": sample_path,
            }
            data_list.append(data_info)
        return data_list
