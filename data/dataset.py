from functools import lru_cache

import ogb
import numpy as np
import torch
from torch.nn import functional as F

from .wrapper import MyPygGraphPropPredDataset
from .collator import collator

from typing import Optional, Union
from torch_geometric.data import Data as PYGDataset
from dgl.data import DGLDataset
from .dgl_datasets import DGLDatasetLookupTable, GraphormerDGLDataset
from .pyg_datasets import PYGDatasetLookupTable, GraphormerPYGDataset
from .ogb_datasets import OGBDatasetLookupTable





class GraphormerDataset:
    def __init__(
        self,
        dataset: Optional[Union[PYGDataset, DGLDataset]] = None,
        dataset_spec: Optional[str] = None,
        dataset_source: Optional[str] = None,
        seed: int = 0,
        train_idx = None,
        valid_idx = None,
        test_idx = None,
    ):
        super().__init__()
        if dataset is not None:
            if dataset_source == "dgl":
                self.dataset = GraphormerDGLDataset(dataset, train_idx, valid_idx, test_idx)
            elif dataset_source == "pyg":
                self.dataset = GraphormerPYGDataset(dataset, train_idx, valid_idx, test_idx)
        elif dataset_source == "dgl":
            self.dataset = DGLDatasetLookupTable.GetDGLDataset(dataset_spec, seed)
        elif dataset_source == "pyg":
            self.dataset = PYGDatasetLookupTable.GetPYGDataset(dataset_spec, seed)
        elif dataset_source == "ogb":
            self.dataset = OGBDatasetLookupTable.GetOGBDataset(dataset_spec, seed)
        self.setup()

    def setup(self):
        self.train_idx = self.dataset.train_idx
        self.valid_idx = self.dataset.valid_idx
        self.test_idx = self.dataset.test_idx

        self.dataset_train = self.dataset.train_data
        self.dataset_val = self.dataset.valid_data
        self.dataset_test = self.dataset.test_data
