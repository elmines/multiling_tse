#!/usr/bin/env python3
"""
Usage ./compute_mean.py metric_1 metric_2 ... metric_n - file_1.csv file_2.csv ...
"""
import csv
import sys
from collections import defaultdict

if __name__ == "__main__":
    i = 1
    metrics = []
    paths = []
    while i < len(sys.argv) and sys.argv[i] != '-':
        metrics.append(sys.argv[i])
        i += 1

    if i < len(sys.argv) and sys.argv[i] == '-':
        i += 1
    while i < len(sys.argv):
        paths.append(sys.argv[i])
        i += 1
    assert paths, "No paths given"

    met_dict = {m:[] for m in metrics} if metrics else defaultdict(list)

    for p in paths:
        with open(p, 'r') as r:
            reader = csv.DictReader(r) 
            row = next(reader)
            for m in metrics or row.keys():
                v = row[m]
                if v is None or v == '':
                    v = 0.0
                met_dict[m].append(float(v))
    met_dict = dict(met_dict)
    met_means = {m:sum(entries)/len(entries) for m,entries in met_dict.items()}
    out_writer = csv.DictWriter(sys.stdout, fieldnames=metrics or met_dict.keys())
    out_writer.writeheader()
    out_writer.writerow(met_means)