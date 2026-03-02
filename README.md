# Target-Stance Extraction

## Repo Structure

- `mtse/`: The main python libary
- `scripts/`: Any script that requires you to have the Python dependencies installed
- `utils/`: Any script that relies solely on bash or the Python STL

## Multilingual Experiments (LREC 2026)

This outlines our experiments for our submission, ``Multilingual Target-Stance Extraction," to LREC 2026.

### Getting the Data

```bash
utils/kptimes_download.sh    # Downloads KPTimes data
utils/multiling_download.sh  # Downloads multilingual stance corpora
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
This simple loop will run the 5-fold cross validation for each of our three target pools (Full, LLM, Manual).
In practice, it's best to run these as 15 separate jobs.
```bash
for fold in {0..4}
do
    SAVE_DIR=./lightning_logs ALL=1                                   scripts/mling_tgen.sh $fold
    SAVE_DIR=./lightning_logs ALL=1 TARGET_TYPE=llm   EXP_NAME=Llm    scripts/mling_tgen.sh $fold
    SAVE_DIR=./lightning_logs ALL=1 TARGET_TYPE=short EXP_NAME=Manual scripts/mling_tgen.sh $fold
done
```


## Dependencies

### Bash

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

## English Experiments

This outlines our one-pass TSE experiments on English data, as well as our reproductions of those two-pass experiments of [Li et al. (2023)](https://aclanthology.org/2023.acl-long.560/).

### Data

First download the KPTimes data
```bash
utils/kptimes_download.sh
```

Download the `raw_(train|val|test)_all_onecol.csv` files from Li et al. (2023)'s [Google Drive](https://drive.google.com/drive/folders/16asK-Ouv6BwXuqUU-J7NwSQS9_k5E4_d)
and copy them to [./data/classic_tse/raw](./data/classic_tse/raw).

### Preprocessing

```bash
# Split the TSE merged corpus into its component corpora (SemEval 2016, P-Stance, etc.).
conda activate ./venv
python scripts/split_tse.py
```

### Execution
This simple loop will run every experiment. In practice, it's better to separate these into different jobs.
```bash
conda activate ./venv
for seed in {0..2}
do
	# Two-Pass TC experiments
	ALL=1                               scripts/multi_li_tcls.sh $seed
	ALL=1 WITH_SE_BUG=1                 scripts/multi_li_tcls.sh $seed
	ALL=1               SCRUB_TARGETS=1 scripts/multi_li_tcls.sh $seed
	ALL=1 WITH_SE_BUG=1 SCRUB_TARGETS=1 scripts/multi_li_tcls.sh $seed

	# One-Pass TC Experiments
	ALL=1                               scripts/multi_oneshot_tcls.sh $seed
	ALL=1 WITH_SE_BUG=1                 scripts/multi_oneshot_tcls.sh $seed
	ALL=1               SCRUB_TARGETS=1 scripts/multi_oneshot_tcls.sh $seed
	ALL=1 WITH_SE_BUG=1 SCRUB_TARGETS=1 scripts/multi_oneshot_tcls.sh $seed

	# Two-Pass TC Experiments
	ALL=1                               scripts/multi_classic_tgen.sh $seed
	ALL=1 WITH_SE_BUG=1                 scripts/multi_classic_tgen.sh $seed

	# One-Pass TG Experiments
	ALL=1                               scripts/multi_oneshot_tgen.sh $seed
	ALL=1 WITH_SE_BUG=1                 scripts/multi_oneshot_tgen.sh $seed

done
```

