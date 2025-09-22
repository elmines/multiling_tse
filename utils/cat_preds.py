#!/usr/bin/env python3
"""
Usage: python cat_preds.py ./path/to/exp_root/ out.csv [seed=0] 

where exp_root/ contains seed{seed}_target_gen/, seed{seed}_target_translate/,
and seed{seed}_target_map/.

"""
import csv
import os
import sys
import glob

def altprint(*args, **kwargs):
    print(f"{sys.argv[0]}:", *args, **kwargs)

if len(sys.argv) not in {3, 4}:
    altprint(__doc__)
    sys.exit(1)

data_dir = os.path.join(os.path.dirname(sys.argv[0]), "..", "data", "multiling")
exp_root = sys.argv[1]
out_path = sys.argv[2]
seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0

gen_dir   = os.path.join(exp_root, f"seed{seed}_target_gen"  )
trans_dir = os.path.join(exp_root, f"seed{seed}_target_translate")
pred_dir  = os.path.join(exp_root, f"seed{seed}_target_map"  )

data_paths = glob.glob(os.path.join(data_dir, "*_val.csv"))
data_parts = [os.path.basename(p).split("_val.csv")[0] for p in data_paths]

def extract_cands(p, k):
    with open(p, 'r') as r:
        trans_rows = []
        last_sid = None
        reader = csv.DictReader(r)
        for row in reader:
            sid = row['Sample']
            targ = row['Generated Target']
            if sid != last_sid:
                trans_rows.append([targ])
                last_sid = sid
            else:
                trans_rows[-1].append(targ)
        trans_rows = [{k: "[" + ';'.join(row) + "]"} for row in trans_rows]
    return trans_rows

catted_rows = []
fieldnames = None
for part_name, data_path in zip(data_parts, data_paths):
    gen_path = os.path.join(gen_dir, f"target_gens.{part_name}.txt")
    trans_path = os.path.join(trans_dir, f"target_gens.{part_name}.txt")
    pred_path = os.path.join(pred_dir, f"target_preds.{part_name}.txt")
    if not os.path.exists(gen_path):
        altprint(f"Skipping {part_name} for missing generations")
        continue
    if not os.path.exists(trans_path):
        altprint(f"Skipping {part_name} for missing translations")
        continue
    if not os.path.exists(pred_path):
        altprint(f"Skipping {part_name} for missing predictions")
        continue

    with open(data_path, 'r') as r:
        data_rows = list(csv.DictReader(r))
    with open(pred_path, 'r') as r:
        pred_rows = list(csv.DictReader(r))
    gen_rows = extract_cands(gen_path, "UntranslatedCandidates")
    trans_rows = extract_cands(trans_path, "TranslatedCandidates")


    if len(data_rows) != len(pred_rows) != len(trans_rows):
        altprint(f"Skipping {part_name} for row inequality")
        continue
    catted_rows.extend( {"Partition": part_name, **ra,  **rb, **rc, **rd} for ra,rb,rc,rd in zip(data_rows, pred_rows, gen_rows, trans_rows)  )

stance_map = {"0": "AGAINST", "1": "FAVOR", "2": "NEUTRAL"}
for row in catted_rows:
    row['Stance'] = stance_map[row['Stance']]

fieldnames = [
    "Partition",
    "Lang",
    "GT Target",
    "Stance",
    "Context",
    "UntranslatedCandidates",
    "TranslatedCandidates",
    "Generated Target",
    "Mapped Target",
]

fieldmap = {
    "Partition": "Partition",
    "Lang": "Lang",
    "GT Target": "GoldTarget",
    "Stance": "GoldStance",
    "Context": "Context",
    "UntranslatedCandidates": "UntranslatedCandidates",
    "TranslatedCandidates": "TranslatedCandidates",
    "Generated Target": "ChosenCandidate",
    "Mapped Target": "MappedTarget"
}

fieldnames = [fieldmap[f] for f in fieldnames]
catted_rows = [{fieldmap[k]:v for k,v in row.items() if k in fieldmap} for row in catted_rows]
    
with open(out_path, 'w') as w:
    writer = csv.DictWriter(w, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(catted_rows)