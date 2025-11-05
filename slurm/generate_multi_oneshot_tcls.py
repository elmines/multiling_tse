#!/usr/bin/env python3
import json
import sys
import os
from itertools import product
from common import write_code

this_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
repo_dir = os.path.join(this_dir, "..")
out_dir = os.path.join(this_dir, "scripts_generate_multi_oneshot_tcls")
os.makedirs(out_dir, exist_ok=True)

with open(os.path.join(this_dir, 'secret.json')) as r:
    secrets = json.load(r)
user_email = secrets["email"]

sbatch_template = """#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --time=1:00:00
#SBATCH --job-name={name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8gb
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
    ("", ""),
    ("_with_scrub", "SCRUB_TARGETS=1"),
    ("_with_bug", "WITH_SE_BUG=1"),
    ("_with_bug_with_scrub", "WITH_SE_BUG=1 SCRUB_TARGETS=1"),
]

for seed, variant  in product(range(3), variants):
    suffix, var_settings = variant
    name = f"multi_oneshot_tcls_seed{seed}{suffix}"
    command = f"{var_settings} scripts/multi_oneshot_tcls.sh {seed}"
    write_code(os.path.join(out_dir, f"{name}.sh"),
        sbatch_template.format(name=name,
                                       command=command,
                                       user_email=user_email,
                                       repo_dir=repo_dir)
    )