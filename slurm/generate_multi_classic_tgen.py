#!/usr/bin/env python3
import sys
import os
from itertools import product
import json

from common import write_code

# Can't hard-code your email in here--don't want that published on GitHub
this_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
repo_dir = os.path.join(this_dir, "..")
out_dir = os.path.join(this_dir, "scripts_generate_multi_classic_tgen")
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
#SBATCH --mem=64gb
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

{command}
"""

variants = [
    ("",           "                 FT_EMBED=1 TARGET_FIT=1 TARGET_PRED=1 TARGET_TEST=1", "8:00:00"),
    ("_with_scrub", "SCRUB_TARGETS=1                         TARGET_PRED=1 TARGET_TEST=1", "2:00:00")
]

for seed, variant in product(range(3), variants):
    suffix, var_settings, duration = variant
    name = f"multi_classic_tgen_seed{seed}_target{suffix}"
    command = f"{var_settings} scripts/multi_classic_tgen.sh {seed}"
    bash_code = target_stage_template.format(name=name,
                                             command=command,
                                             user_email=user_email,
                                             repo_dir=repo_dir,
                                             partition=b200_part,
                                             duration=duration)
    write_code(os.path.join(out_dir, f"{name}.sh"), bash_code)

stance_stage_template = """#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
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

{command}
"""


variants = [
    ("",           "" ),
    ("_with_scrub", "SCRUB_TARGETS=1" )
]

for seed, variant  in product(range(3), variants):
    suffix, var_settings = variant

    name = f"multi_classic_tgen_seed{seed}_stance{suffix}"
    command = f"{var_settings} STANCE_FIT=1 STANCE_TEST=1 AGG_PRED=1 TSE_TEST=1 GT_TSE_TEST=1 scripts/multi_classic_tgen.sh {seed}"
    bash_code = stance_stage_template.format(name=name,
                                       command=command,
                                       user_email=user_email,
                                       repo_dir=repo_dir)
    write_code(os.path.join(out_dir, f"{name}.sh"), bash_code)