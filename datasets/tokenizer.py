"""SMILES-based tokenization utilities.
"""

__all__ = ("PAD_TOKEN", "BOS_TOKEN", "EOS_TOKEN", "UNK_TOKEN", "SUFFIX", "CLS_TOKEN", "START_TOKEN", "SEP_TOKEN",
           "SPECIAL_TOKENS", "PAD_TOKEN_ID", "BOS_TOKEN_ID", "EOS_TOKEN_ID",
           "UNK_TOKEN_ID", "SMILESBPETokenizer", "SMILESAlphabet")


from collections.abc import Collection, Iterator
from dataclasses import dataclass
from itertools import chain
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple, Union
from tokenizers import AddedToken, Tokenizer
from tokenizers import decoders, models, normalizers, processors, trainers
from tokenizers.implementations import BaseTokenizer
from transformers import PreTrainedTokenizerFast


SUFFIX, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, CLS_TOKEN, START_TOKEN, SEP_TOKEN = "", "<pad>", "<s>", "</s>", "<unk>", "<cls>", "<start>", "<sep>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN, CLS_TOKEN, START_TOKEN, SEP_TOKEN]
PAD_TOKEN_ID, BOS_TOKEN_ID, EOS_TOKEN_ID, UNK_TOKEN_ID, CLS_TOKEN_ID, START_TOKEN_ID, SEP_TOKEN_ID = range(7)


class SMILESBPETokenizer(BaseTokenizer):
    """Tokenizes SMILES strings and applies BPE.

    Args:
        vocab (`str` or `dict`, optional, defaults to `None`):
            Token vocabulary.
        merges (`str` or `dict` or `tuple`, optional, defaults to `None`):
            BPE merges.
        unk_token (`str` or `tokenizers.AddedToken`, optional, defaults to "<unk>")
        suffix (`str`, defaults to "")
        dropout (`float`, defaults to `None`)

    Examples:
        >>> tokenizer = SMILESBPETokenizer()
        >>> tokenizer.train("path-to-smiles-strings-file")
        Tokenization logs...
        >>> tokenizer.save_model("checkpoints-path")
        >>> same_tokenizer = SMILESBPETokenizer.from_file("checkpoints-path/vocab.json",
        ...                                               "checkpoints-path/merges.txt")
    """

    def __init__(
        self,
        vocab: Optional[Union[str, Dict[str, int]]] = None,
        merges: Optional[Union[str, Dict[Tuple[int, int], Tuple[int, int]]]] = None,
        unk_token: Union[str, AddedToken] = "<unk>",
        suffix: str = SUFFIX,
        dropout: Optional[float] = None,
    ) -> None:
        unk_token_str = str(unk_token)

        tokenizer = Tokenizer(models.BPE(vocab, merges, dropout=dropout,
                                         unk_token=unk_token_str,
                                         end_of_word_suffix=suffix))

        if tokenizer.token_to_id(unk_token_str) is not None:
            tokenizer.add_special_tokens([unk_token_str])

        tokenizer.normalizer = normalizers.Strip(left=False, right=True)
        tokenizer.decoder = decoders.Metaspace(add_prefix_space=True)
        tokenizer.post_processor = processors.TemplateProcessing(
            single=f"{BOS_TOKEN} $A {EOS_TOKEN}",
            special_tokens=[(BOS_TOKEN, BOS_TOKEN_ID), (EOS_TOKEN, EOS_TOKEN_ID)])

        parameters = {"model": "BPE", "unk_token": unk_token, "suffix": suffix,
                      "dropout": dropout}

        super().__init__(tokenizer, parameters)

    @classmethod
    def from_file(cls, vocab_filename: str, merges_filename: str, **kwargs) \
            -> "SMILESBPETokenizer":
        vocab, merges = models.BPE.read_file(vocab_filename, merges_filename)
        return cls(vocab, merges, **kwargs)

    def train(
        self,
        files: Union[str, List[str]],
        vocab_size: int = 1_000,
        min_frequency: int = 2,
        special_tokens: List[Union[str, AddedToken]] = None,
        limit_alphabet: int = 200,
        initial_alphabet: List[str] = None,
        suffix: Optional[str] = SUFFIX,
        show_progress: bool = True,
    ) -> None:
        special_tokens = special_tokens or SPECIAL_TOKENS
        initial_alphabet = initial_alphabet or []

        trainer = trainers.BpeTrainer(vocab_size=vocab_size,
                                      min_frequency=min_frequency,
                                      special_tokens=special_tokens,
                                      limit_alphabet=limit_alphabet,
                                      initial_alphabet=initial_alphabet,
                                      end_of_word_suffix=suffix,
                                      show_progress=show_progress)
        if isinstance(files, str):
            files = [files]
        self._tokenizer.train(files, trainer=trainer)

    def train_from_iterator(
        self,
        iterator: Iterator,
        vocab_size: int = 1_000,
        min_frequency: int = 2,
        special_tokens: List[Union[str, AddedToken]] = None,
        limit_alphabet: int = 200,
        initial_alphabet: List[str] = None,
        suffix: Optional[str] = SUFFIX,
        show_progress: bool = True,
    ) -> None:
        special_tokens = special_tokens or SPECIAL_TOKENS
        initial_alphabet = initial_alphabet or []

        trainer = trainers.BpeTrainer(vocab_size=vocab_size,
                                      min_frequency=min_frequency,
                                      special_tokens=special_tokens,
                                      limit_alphabet=limit_alphabet,
                                      initial_alphabet=initial_alphabet,
                                      end_of_word_suffix=suffix,
                                      show_progress=show_progress)
        self._tokenizer.train_from_iterator(iterator, trainer=trainer)

    @staticmethod
    def get_hf_tokenizer(
        tokenizer_file: str,
        special_tokens: Optional[Dict[str, str]] = None,
        model_max_length: int = 512,
        *init_inputs, **kwargs
    ) -> PreTrainedTokenizerFast:
        """Gets HuggingFace tokenizer from the pretrained `tokenizer_file`. Optionally,
        appends `special_tokens` to vocabulary and sets `model_max_length`.
        """
        tokenizer = PreTrainedTokenizerFast(tokenizer_file=tokenizer_file,
                                            *init_inputs, **kwargs)
        special_tokens = special_tokens or dict(zip(
            ["pad_token", "bos_token", "eos_token", "unk_token"],
            SPECIAL_TOKENS))
        tokenizer.add_special_tokens(special_tokens)
        tokenizer.model_max_length = model_max_length
        return tokenizer


@dataclass(init=True, eq=False, repr=True, frozen=True)
class SMILESAlphabet(Collection):
    atoms: FrozenSet[str] = frozenset([
        'Ac', 'Ag', 'Al', 'Am', 'Ar', 'As', 'At', 'Au', 'B', 'Ba', 'Be', 'Bh',
        'Bi', 'Bk', 'Br', 'C', 'Ca', 'Cd', 'Ce', 'Cf', 'Cl', 'Cm', 'Co', 'Cr',
        'Cs', 'Cu', 'Db', 'Dy', 'Er', 'Es', 'Eu', 'F', 'Fe', 'Fm', 'Fr', 'Ga',
        'Gd', 'Ge', 'H', 'He', 'Hf', 'Hg', 'Ho', 'Hs', 'I', 'In', 'Ir', 'K',
        'Kr', 'La', 'Li', 'Lr', 'Lu', 'Md', 'Mg', 'Mn', 'Mo', 'Mt', 'N', 'Na',
        'Nb', 'Nd', 'Ne', 'Ni', 'No', 'Np', 'O', 'Os', 'P', 'Pa', 'Pb', 'Pd',
        'Pm', 'Po', 'Pr', 'Pt', 'Pu', 'Ra', 'Rb', 'Re', 'Rf', 'Rh', 'Rn',
        'Ru', 'S', 'Sb', 'Sc', 'Se', 'Sg', 'Si', 'Sm', 'Sn', 'Sr', 'Ta', 'Tb',
        'Tc', 'Te', 'Th', 'Ti', 'Tl', 'Tm', 'U', 'V', 'W', 'Xe', 'Y', 'Yb',
        'Zn', 'Zr'
    ])

    # Bonds, charges, etc.
    non_atoms: FrozenSet[str] = frozenset([
        '-', '=', '#', ':', '(', ')', '.', '[', ']', '+', '-', '\\', '/', '*',
        '1', '2', '3', '4', '5', '6', '7', '8', '9', '0',
        '@', 'AL', 'TH', 'SP', 'TB', 'OH',
    ])

    additional: FrozenSet[str] = frozenset()

    def __contains__(self, item: Any) -> bool:
        return item in self.atoms or item in self.non_atoms

    def __iter__(self):
        return (token for token in chain(self.atoms, self.non_atoms))

    def __len__(self) -> int:
        return len(self.atoms) + len(self.non_atoms) + len(self.additional)

    def get_alphabet(self) -> Set[str]:
        alphabet = set()
        for token in self.atoms:
            if len(token) > 1:
                alphabet.update(list(token))
                alphabet.add(token[0].lower())
            else:
                alphabet.add(token)
                alphabet.add(token.lower())
        for token in chain(self.non_atoms, self.additional):
            if len(token) > 1:
                alphabet.update(list(token))
            else:
                alphabet.add(token)
        return alphabet
