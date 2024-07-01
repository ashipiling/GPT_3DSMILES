"""Loads torch-compatible data sets and lightning-compatible data modules.
"""

__all__ = ("CSVDataset", "CSVDataModule", "CVSplitter", "LMDataset", "LMDataModule")

from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple, Union

import torch
from pytorch_lightning import LightningDataModule
from sklearn.model_selection import ShuffleSplit
from tokenizers.implementations import BaseTokenizer
from transformers import PreTrainedTokenizerFast
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding
from torch.utils.data import Dataset, DataLoader


@dataclass(init=True, repr=True, eq=False, frozen=False)
class CSVDataset(Dataset):
    """Stores `pandas.DataFrame` instance of tabular data and retrieves encoded token
    ids and attention mask. Optionally returns labels and their masks.

    Args:
        dataframe (`pandas.DataFrame`):
            Data frame of SMILES strings and their (multi-task) labels.
        tokenizer (`tokenizers.BaseTokenizer` or `SMILESBPETokenizer`)
            SMILES tokenizer.
        smiles_column (`str`, defaults to "smiles"):
            Column name of SMILES strings in `dataframe`.
        target_column (`str` or `list` of `str`, defaults to `None`):
            Target column(s). If `None`, labels are ignored.
        has_empty_target (`bool`, defaults to `False`):
            Whether entries have empty target values. If `True`, additionally retrieves
            a target mask.
        task_type ("classification" or "regression", defaults to "classification")
        encode_kwargs (dict, defaults to {"truncation": True})
            Positional arguments for `tokenizer` encoding, e.g. {"padding": True}.
    """

    dataframe: "pandas.DataFrame"
    tokenizer: BaseTokenizer
    smiles_column: str = 'smiles'
    target_column: Union[None, str, List[str]] = None
    has_empty_target: bool = False
    task_type: Literal["classification", "regression"] = "classification"
    encode_kwargs: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        if isinstance(self.tokenizer, PreTrainedTokenizerFast):
            self._encode = partial(self.tokenizer.__call__, add_special_tokens=False)
            self._id_key = "input_ids"
        else:
            self._encode = self.tokenizer.encode
            self._id_key = "ids"
        self.encode_kwargs = self.encode_kwargs or {"truncation": True}
        self._encode = partial(self._encode, **self.encode_kwargs)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """Returns dict of encoded token IDs, attention mask, and optionally labels
        and label mask.
        """
        item: Dict[str, torch.Tensor] = {}

        smiles = self.dataframe.iloc[index][self.smiles_column]
        encodings = self._encode(smiles)
        item["input_ids"] = torch.LongTensor(getattr(encodings, self._id_key))
        item["attention_mask"] = torch.LongTensor(getattr(encodings, "attention_mask"))

        if self.target_column is not None:
            labels = self.dataframe.iloc[index][self.target_column]
            if self.has_empty_target:
                label_mask = ~labels.isna()
                labels = labels.fillna(-1)
                item["label_mask"] = torch.BoolTensor(label_mask)
            if self.task_type == "regression":
                tensor_type = torch.FloatTensor
            elif self.task_type == "classification":
                tensor_type = torch.LongTensor
            else:
                raise NotImplementedError("`CSVDataset` supports only classification and "
                                          "regression tasks")
            item["labels"] = tensor_type(labels)

        return item

    def __len__(self) -> int:
        return self.dataframe.shape[0]


@dataclass(init=True, eq=True, repr=True, frozen=False)
class CVSplitter:
    """Splits series of SMILES data with either random or scaffold splitting.
    """

    mode: str = "random"
    train_size: float = 0.8
    val_size: float = 0.1
    test_size: float = 0.1

    def __post_init__(self) -> None:
        if self.mode == "scaffold":
            self.train_val_test_split = self.scaffold_split
        elif self.mode == "random":
            self.train_val_test_split = self.random_split

    @staticmethod
    def get_sorted_scaffolds(smiles_seqs: Sequence[str]):
        from rdkit.Chem import MolFromSmiles
        from rdkit.Chem.Scaffolds.MurckoScaffold import MurckoScaffoldSmiles

        scaffolds: Dict[str, List[int]] = defaultdict(list)
        molecules = (MolFromSmiles(s, sanitize=True) for s in smiles_seqs)

        for i, molecule in enumerate(molecules):
            try:
                scaffold = MurckoScaffoldSmiles(mol=molecule, includeChirality=False)
                scaffolds[scaffold].append(i)
            except Exception:  # Really don't know what exception is raised...
                pass

        scaffolds = {scaffold: sorted(ids) for scaffold, ids in scaffolds.items()}
        scaffold_sets = [scaffold_set
                         for scaffold, scaffold_set in
                         sorted(scaffolds.items(), key=lambda x: (len(x[1]), x[1][0]),
                                reverse=True)]
        return scaffold_sets

    def scaffold_split(self, smiles_seqs: Sequence[str]) \
            -> Tuple[List[int], List[int], List[int]]:
        scaffold_sets = self.get_sorted_scaffolds(smiles_seqs)

        n_samples = len(smiles_seqs)
        train_idx, val_idx, test_idx = [], [], []
        train_cutoff = int(self.train_size * n_samples)
        val_cutoff = int((self.train_size + self.val_size) * n_samples)

        for group_indices in scaffold_sets:
            n_group = len(group_indices)
            n_train = len(train_idx)
            if n_train + n_group > train_cutoff:
                n_val = len(val_idx)
                if n_train + n_val + n_group > val_cutoff:
                    test_idx.extend(group_indices)
                else:
                    val_idx.extend(group_indices)
            else:
                train_idx.extend(group_indices)

        return train_idx, val_idx, test_idx

    def random_split(self, smiles_seqs: "pandas.Series") \
            -> Tuple["numpy.array", "numpy.array", "numpy.array"]:
        cv = ShuffleSplit(train_size=self.train_size + self.val_size)
        train_idx, val_idx = next(cv.split(smiles_seqs))
        cv.train_size = 1 - self.test_size / (self.train_size + self.val_size)
        train_idx, test_idx = next(cv.split(smiles_seqs.iloc[train_idx]))

        return train_idx, val_idx, test_idx


@dataclass(init=True, repr=True, eq=False, frozen=False)
class CSVDataModule(LightningDataModule):
    """Lightning data module for tabular data. Accepts pandas `dataframe`, splits the
    data into train/valid/test with `splitter`, creates `CSVDataset`s and Pytorch
    `DataLoader`s with `DataCollatorWithPadding` collate function.
    """

    dataframe: "pandas.DataFrame"
    tokenizer: BaseTokenizer
    smiles_column: str = "smiles"
    target_column: Union[None, str, List[str]] = None
    has_empty_target: bool = False
    task_type: Literal["classification", "regression"] = "classification"
    splitter: CVSplitter = CVSplitter()
    batch_size: int = 16
    num_workers: int = 0

    def __post_init__(self) -> None:
        super().__init__()
        self.train_dataset: Optional[CSVDataset] = None
        self.val_dataset: Optional[CSVDataset] = None
        self.test_dataset: Optional[CSVDataset] = None
        self.collate_fn: Callable = DataCollatorWithPadding(self.tokenizer)

    def setup(self, stage: Optional[str] = None) -> None:
        train_idx, val_idx, test_idx = self.splitter.train_val_test_split(
            self.dataframe[self.smiles_column])

        train_dataframe = self.dataframe.iloc[train_idx].reset_index(drop=True)
        self.train_dataset = CSVDataset(train_dataframe, self.tokenizer,
                                        self.smiles_column, self.target_column,
                                        self.has_empty_target, self.task_type)
        valid_dataframe = self.dataframe.iloc[val_idx].reset_index(drop=True)
        self.val_dataset = CSVDataset(valid_dataframe, self.tokenizer,
                                      self.smiles_column, self.target_column,
                                      self.has_empty_target, self.task_type)
        test_dataframe = self.dataframe.iloc[test_idx].reset_index(drop=True)
        self.test_dataset = CSVDataset(test_dataframe, self.tokenizer,
                                       self.smiles_column, self.target_column,
                                       self.has_empty_target, self.task_type)

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader],
                                        Dict[str, DataLoader]]:
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True,
                          collate_fn=self.collate_fn, num_workers=self.num_workers)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader],
                                      Dict[str, DataLoader]]:
        return DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False,
                          collate_fn=self.collate_fn, num_workers=self.num_workers)

    def test_dataloader(self) -> Union[DataLoader, List[DataLoader],
                                       Dict[str, DataLoader]]:
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False,
                          collate_fn=self.collate_fn, num_workers=self.num_workers)


@dataclass(init=True, eq=False, repr=True, frozen=False)
class LMDataset(Dataset):
    """Simple sequential dataset for autoregressive language modeling.
    """

    filename: str
    tokenizer: BaseTokenizer

    def __post_init__(self) -> None:
        self.smiles_strings = Path(self.filename).read_text(encoding='ascii').splitlines()

        if isinstance(self.tokenizer, PreTrainedTokenizerFast):
            self._encode = partial(self.tokenizer.__call__, truncation=True)
            self._id_key = "input_ids"
        else:
            self._encode = self.tokenizer.encode
            self._id_key = "ids"

    def __len__(self) -> int:
        return len(self.smiles_strings)

    def __getitem__(self, i: int) -> torch.Tensor:
        encodings = self._encode(self.smiles_strings[i])
        return torch.LongTensor(getattr(encodings, self._id_key))


@dataclass(init=True, repr=True, eq=False, frozen=False)
class LMDataModule(LightningDataModule):
    """Lightning data module for autoregressive language modeling.
    """

    filename: str
    tokenizer: BaseTokenizer
    batch_size: int = 128
    num_workers: int = 0
    collate_fn: Union[None, Literal["default"], Callable] = "default"

    def __post_init__(self) -> None:
        super().__init__()
        if self.collate_fn == "default":
            self.collate_fn = DataCollatorForLanguageModeling(self.tokenizer, mlm=False)

    def setup(self, stage: Optional[str] = None) -> None:
        self.dataset = LMDataset(self.filename, self.tokenizer)

    def train_dataloader(self) -> Union[DataLoader, List[DataLoader],
                                        Dict[str, DataLoader]]:
        return DataLoader(self.dataset, batch_size=self.batch_size, shuffle=True,
                          collate_fn=self.collate_fn, num_workers=self.num_workers)
