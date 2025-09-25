# multiling_tse

## Repo Structure

`mtse/`: The main python libary
`scripts/`: Any script that requires you to have the dependencies installed
`utils/`: Any script that doesn't require anything beyond bash and a Python3 interpreter

## Dependencies

The easiest approach is to make a conda environment:
```bash
conda env create -f environment.yaml --prefix ./venv
```

However, we did also make the `mtse/` module `pip`-installable if you're using it in another project:
```bash
python -m pip install .
```

## Common Instructions

### Data Downloads

```bash
utils/kptimes_download.sh
```

## Multilingual Experiments

### Getting the Data

```bash
utils/multiling_download.sh
```
The script will give you two password prompts for the Sardistance dataset.
To get the passwords, request access from the organizers [here](https://forms.gle/xuikYEsHB18uVVQ67).

### Preprocessing

```bash
utils/multiling_preproc.sh
# trans_kptimes.sh uses a HuggingFace model,
# so we need the conda environment here
conda activate ./venv
scripts/trans_kptimes.sh
```

## English Experiments

### Getting the Data

Download the `raw_(train|val|test)_all_onecol.csv` files from their [Google Drive](https://drive.google.com/drive/folders/16asK-Ouv6BwXuqUU-J7NwSQS9_k5E4_d)
and copy them to [./data/li_tse](./data/li_tse).

### Preprocessing
Run `python3 scripts/split_tse.py` to split the merged TSE corpus into its component corpora (SemEval 2016, P-Stance, etc.).
