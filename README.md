# multiling_tse

# Getting the Data

## Data from Li et al. (2023)
Download the `raw_(train|val|test)_all_onecol.csv` files from their [Google Drive](https://drive.google.com/drive/folders/16asK-Ouv6BwXuqUU-J7NwSQS9_k5E4_d)
and copy them to [./data/li_tse](./data/li_tse).

Run `python3 scripts/split_tse.py` to split the merged TSE corpus into its component corpora (SemEval 2016, P-Stance, etc.).

## Keyword Generation Training Data

```bash
cd data/kp_times
./download.sh
```

## Repo Structure

`scripts/`: Any script that requires you to have the dependencies installed
`utils/`: Any script that doesn't require anything beyond bash and a Python3 interpreter