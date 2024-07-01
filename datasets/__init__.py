from .pocket_datasets_2 import FragSmilesPocketPretrainDataset


def build_datasets(cfg, mode):
    return FragSmilesPocketPretrainDataset.build_datasets(cfg, mode)

