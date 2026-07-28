from abc import ABCMeta, abstractmethod
from typing import Dict, Optional

import torch
import numpy as np

class BaseTransform(metaclass=ABCMeta):
    """Base class for all transformations."""

    def __call__(self, results: Dict) -> Optional[Dict]:
        return self.transform(results)

    @abstractmethod
    def transform(self, results: Dict) -> Optional[Dict]:
        pass


class LoadNpzFile(BaseTransform):
    """Load an .npz file containing pair embeddings.

    Required keys: sample_path
    Added keys: Z_II (L, L, 128)
    """

    def transform(self, results: Dict) -> Dict:
        sample_path = results["sample_path"]
        
        with np.load(sample_path) as data:
            pair_embeddings = data["pair_embeddings"]
            
        # print(f"loading {sample_path}, shape={pair_embeddings.shape}")
        
        results["Z_II"] = torch.from_numpy(pair_embeddings).float()
        return results

class PackCompressorInputs(BaseTransform):
    """Pack ``Z_II`` into the dictionary expected by the model."""

    def transform(self, results: Dict) -> Dict:
        return {
            "Z_II": results["Z_II"],
        }
