# Multilingual Target-Stance Extraction

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


## Getting the Data

```bash
utils/kptimes_download.sh
utils/multiling_download.sh
```
The script will give you two password prompts for the Sardistance dataset.
To get the passwords, request access from the organizers [here](https://forms.gle/xuikYEsHB18uVVQ67).

## Preprocessing

```bash
utils/preproc_mling.py
utils/part_kptimes.py
# trans_kptimes.sh uses a HuggingFace model,
# so we need the conda environment here
conda activate ./venv
scripts/trans_kptimes.sh
```

## Execution
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

