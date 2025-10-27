#!/usr/bin/env python3
import sys
import os
import stat
from itertools import product

# Can't hard-code your email in here--don't want that published on GitHub
user_email = sys.argv[1]
sbatch_template = """#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --time=2:00:00
#SBATCH --job-name={name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16gb
#SBATCH --mail-user={user_email}
#SBATCH --mail-type=FAIL,END
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

export XDG_RUNTIME_DIR=$SLURM_TMPDIR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
date
hostname
script_dir=$(dirname $0)
cd $script_dir/..
pwd

source "/apps/conda/25.3.1/etc/profile.d/conda.sh"
git log -1
conda activate ./venv

ALL=1 {command}
"""

variants = [
    (""                    , "                             "),
    ("_with_bug_with_scrub", "WITH_SE_BUG=1 SCRUB_TARGETS=1"),
    ("_with_bug"           , "WITH_SE_BUG=1                "),
    ("_with_scrub"         , "              SCRUB_TARGETS=1"),
]

for seed, variant  in product(range(3), variants):
    suffix, var_settings = variant
    name = f"multi_li_tcls_seed{seed}{suffix}"
    command = f"{var_settings} scripts/multi_li_tcls.sh {seed}"
    bash_code = sbatch_template.format(name=name, command=command, user_email=user_email)
    out_path = f"{name}.sh"
    with open(out_path, 'w') as w:
        w.write(bash_code)
    # All this just to do the equivalent of `chmod +x` ...
    os.chmod(out_path, os.stat(out_path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

