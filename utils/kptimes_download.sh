#!/bin/bash

pushd $(dirname $0)/../data/kptimes
curl -L -o dev.jsonl https://huggingface.co/datasets/taln-ls2n/kptimes/resolve/main/dev.jsonl?download=true
curl -L -o train.jsonl https://huggingface.co/datasets/taln-ls2n/kptimes/resolve/main/train.jsonl?download=true
popd