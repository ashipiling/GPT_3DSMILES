# GPT_3DSMILES

3DSMILES-GPT: 3D Molecular Pocket-based Generation with Token-only Large Language Model


## Installation

Set up [conda](https://conda.io/en/latest/index.html) and create a new environment from
`environment.yml` (if needed, make corresponding edits for GPU-compatibility).
```shell
conda env create -f environment.yml
conda activate gpt3dsmiles
git clone https://github.com/ashipiling/GPT_3DSMILES.git
cd GPT_3DSMILES
```


## Checkpoints

checkpoints download from (https://drive.google.com/file/d/11iB2MBK4CRC7XIZqVwN0AXZR_RV3tf30/view?usp=sharing)

Unzip and place in the (https://github.com/ashipiling/GPT_3DSMILES/tree/master/checkpoints/) directory.


## Data

PubChem-10M download from (https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/pubchem_10m.txt.zip).

Crossdocked_pocket10 download from (https://github.com/pengxingang/Pocket2Mol/blob/main/data/README.md).

Unzip and place in the (https://github.com/ashipiling/GPT_3DSMILES/tree/master/data/) directory.


## Examples

```shell
python3 sample_from_ligand_msms.py --task_name test_parse_args --pocket_p data/5p9j_blank.pdb --ligand_p data/5p9j.sdf
```

## Citation

