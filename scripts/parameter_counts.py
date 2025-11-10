#!/usr/bin/env python3
# STL
import sys
import os
# 3rd Party
import torch
# Local
from mtse.modules import ClassicTargetGenerator, ClassicStanceClassifierModule, TGOneShotModule, ClassicTargetClassifierModule, OneShotTClsModule

scenario = sys.argv[1]

def get_counts(module: torch.nn.Module):
    trainable = 0
    all = 0
    for p in module.train().parameters():
        numel = p.numel()
        all += numel
        if p.requires_grad:
            trainable += numel
    return trainable, all

TARGETS_PATH = os.path.join(os.path.dirname(sys.argv[0]), "..", "static", "classic_merged_targets.txt")

def get_stance_bert_params():
    train_count, all_count = get_counts(ClassicStanceClassifierModule(
        targets_path=TARGETS_PATH, stance_type='tri'
    ))
    print(f"StanceBERT: trainable={train_count}, all={all_count}")

if scenario == "twoshot_tgen":
    print("Two-shot TG")
    print("-----------")
    train_count, all_count = get_counts(ClassicTargetGenerator(
        TARGETS_PATH, multilingual=False
    ))
    print(f"BART: trainable={train_count}, all={all_count}")
    get_stance_bert_params()
elif scenario == "oneshot_tgen":
    # Only need this for the constructor to work--we don't even count the FT embeddings
    embeddings_path = sys.argv[2] 
    train_count, all_count = get_counts(TGOneShotModule(embeddings_path, TARGETS_PATH, 'tri'))
    print("One-shot TG")
    print("-----------")
    print(f"BART: trainable={train_count}, all={all_count}")
elif scenario == "twoshot_tcls":
    print("Two-shot TC")
    print("-----------")
    train_count, all_count = get_counts(ClassicTargetClassifierModule(targets_path=TARGETS_PATH))
    print(f"TargetBERT: trainable={train_count}, all={all_count}")
    get_stance_bert_params()
else:
    assert scenario == "oneshot_tcls"
    print("One-shot TC")
    print("-----------")
    train_count, all_count = get_counts(OneShotTClsModule(TARGETS_PATH, 'tri'))
    print(f"BERT: trainable={train_count}, all={all_count}")
