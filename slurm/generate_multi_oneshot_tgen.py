#!/usr/bin/env python3
import sys
import os
from itertools import product
import json
from common import write_code

this_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
repo_dir = os.path.join(this_dir, "..")
out_dir = os.path.join(this_dir, "scripts_generate_multi_oneshot_tgen")
os.makedirs(out_dir, exist_ok=True)

with open(os.path.join(this_dir, 'secret.json')) as r:
    secrets = json.load(r)
b200_part = secrets["b200_partition"]
user_email = secrets["email"]


target_stage_template = """#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --partition={partition}
#SBATCH --time={duration}
#SBATCH --job-name={name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24gb
#SBATCH --mail-user={user_email}
#SBATCH --mail-type=FAIL,END
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

export XDG_RUNTIME_DIR=$SLURM_TMPDIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
date
hostname
cd {repo_dir}
pwd

source "/apps/conda/25.3.1/etc/profile.d/conda.sh"
git log -1
conda activate ./venv

ALL=1 {command}
"""

variants = [
    ("", "", "4:00:00"),
    ("_with_scrub", "SCRUB_TARGETS=1 FT_EMBED=0", "2:00:00"),
    ("_with_bug",            "WITH_SE_BUG=1 FT_EMBED=0", "2:00:00"),
    ("_with_bug_with_scrub", "WITH_SE_BUG=1 SCRUB_TARGETS=1 FT_EMBED=0", "2:00:00")
]

for seed, variant in product(range(3), variants):
    suffix, var_settings, duration = variant
    name = f"multi_oneshot_tgen_seed{seed}{suffix}"
    command = f"{var_settings} scripts/multi_oneshot_tgen.sh {seed}"
    write_code(os.path.join(out_dir, f"{name}.sh"),
        target_stage_template.format(name=name,
                                                 duration=duration,
                                                 command=command,
                                                 user_email=user_email,
                                                 repo_dir=repo_dir,
                                                 partition=b200_part)
    )