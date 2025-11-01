#!/usr/bin/env python3
import sys
import os
from itertools import product
import json

from common import write_code

# Can't hard-code your email in here--don't want that published on GitHub
this_dir = os.path.abspath(os.path.dirname(sys.argv[0]))
repo_dir = os.path.join(this_dir, "..")
out_dir = os.path.join(this_dir, "scripts_generate_mling_tgen")
os.makedirs(out_dir, exist_ok=True)

with open(os.path.join(this_dir, 'secret.json')) as r:
    secrets = json.load(r)
b200_part = secrets["b200_partition"]
user_email = secrets["email"]

embed_stage_template ="""#!/bin/bash

#SBATCH --time=3:00:00
#SBATCH --job-name={name}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --mail-user={user_email}
#SBATCH --mail-type=FAIL,END
#SBATCH --output=%x.%j.out
#SBATCH --error=%x.%j.err

export XDG_RUNTIME_DIR=$SLURM_TMPDIR
date
hostname
cd {repo_dir}
pwd

source "/apps/conda/25.3.1/etc/profile.d/conda.sh"
git log -1
conda activate ./venv
{command}
"""

for seed in range(3):
    name = f"mling_tgen_ftembed_seed{seed}"
    command = f"FT_EMBED=1 scripts/mling_tgen.sh N/A {seed}"
    bash_code = embed_stage_template.format(name=name,
                                            command=command,
                                            user_email=user_email,
                                            repo_dir=repo_dir
                                            )
    write_code(os.path.join(out_dir, f"{name}.sh"), bash_code)

target_state_template = """#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --partition={partition}
#SBATCH --time=32:00:00
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

TARGET_FIT=1 {var_settings} scripts/mling_tgen.sh N/A {seed}
for fold in 0 1 2 3 4
do
    TARGET_GEN=1 TARGET_TRANS=1 TARGET_MAP=1 TARGET_TEST=1 {var_settings} scripts/mling_tgen.sh $fold {seed}
done
"""

stance_stage_template = """#!/bin/bash

#SBATCH --gres=gpu:1
#SBATCH --time=15:00:00
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
cd {repo_dir}
pwd

source "/apps/conda/25.3.1/etc/profile.d/conda.sh"
git log -1
conda activate ./venv

for fold in 0 1 2 3 4
do
    STANCE_FIT=1 STANCE_TEST=1 TSE_TEST=1 GT_TSE_TEST=1 {var_settings} scripts/mling_tgen.sh $fold {seed}
done
"""

variants = [
    ("",""),
    ("_short","TARGET_TYPE=short"),
    ("_llm","TARGET_TYPE=llm"),
]

for seed, variant in product(range(3), variants):
    suffix, var_settings = variant
    name = f"mling_tgen_seed{seed}_target{suffix}"
    write_code(os.path.join(out_dir, f"{name}.sh"),
               target_state_template.format(name=name,
                                            seed=seed,
                                            var_settings=var_settings,
                                             user_email=user_email,
                                             repo_dir=repo_dir,
                                             partition=b200_part))
    name = f"mling_tgen_seed{seed}_stance{suffix}"
    write_code(os.path.join(out_dir, f"{name}.sh"),
               stance_stage_template.format(name=name,
                                            seed=seed,
                                            var_settings=var_settings,
                                             user_email=user_email,
                                             repo_dir=repo_dir))

