# Multilingual Target Stance Extraction

## Repo Structure

- `mtse/`: The main python libary
- `scripts/`: Any script that requires you to have the Python dependencies installed
- `utils/`: Any script that doesn't require anything beyond bash and a Python3 interpreter

## Dependencies

### System

- `sudo apt-get install curl wget`

### Python

The easiest approach is to make a conda environment:
```bash
conda env create -f environment.yaml --prefix ./venv
```

However, we did also make the `mtse/` module `pip`-installable if you're using it in another project:
```bash
python -m pip install .
```

## Multilingual Experiments

### Getting the Data

```bash
utils/kptimes_download.sh
utils/multiling_download.sh
```
The script will give you two password prompts for the Sardistance dataset.
To get the passwords, request access from the organizers [here](https://forms.gle/xuikYEsHB18uVVQ67).

### Preprocessing

```bash
utils/preproc_mling.py
utils/part_kptimes.py
# trans_kptimes.sh uses a HuggingFace model,
# so we need the conda environment here
conda activate ./venv
scripts/trans_kptimes.sh
```

### Execution
In theory this will run the 5-fold cross validation for each of our three target pools (Full, LLM, Manual).
In practice, it's best to run these as separate jobs.
```bash
for fold in {0..4}
do
    SAVE_DIR=./lightning_logs ALL=1                                   scripts/mling_tgen.sh $fold
    SAVE_DIR=./lightning_logs ALL=1 TARGET_TYPE=llm   EXP_NAME=Llm    scripts/mling_tgen.sh $fold
    SAVE_DIR=./lightning_logs ALL=1 TARGET_TYPE=short EXP_NAME=Manual scripts/mling_tgen.sh $fold
done
```

## English Experiments

These are not for LREC 2026.

### Data

First download the KPTimes data
```bash
utils/kptimes_download.sh
```

Download the `raw_(train|val|test)_all_onecol.csv` files from Li et al. (2023)'s [Google Drive](https://drive.google.com/drive/folders/16asK-Ouv6BwXuqUU-J7NwSQS9_k5E4_d)
and copy them to [./data/li_tse](./data/li_tse).

### Preprocessing
Run `python3 scripts/split_tse.py` to split the merged TSE corpus into its component corpora (SemEval 2016, P-Stance, etc.).
