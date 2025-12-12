# Target-Stance Extraction

## Repo Structure

- `mtse/`: The main python libary
- `scripts/`: Any script that requires you to have the Python dependencies installed
- `utils/`: Any script that relies solely on bash or the Python STL

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

## Data

First download the KPTimes data
```bash
utils/kptimes_download.sh
```

Download the `raw_(train|val|test)_all_onecol.csv` files from Li et al. (2023)'s [Google Drive](https://drive.google.com/drive/folders/16asK-Ouv6BwXuqUU-J7NwSQS9_k5E4_d)
and copy them to [./data/classic_tse/raw](./data/classic_tse/raw).

## Preprocessing

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

