#!/usr/bin/env python3
"""
Usage ./compute_mean.py metric_1 metric_2 ... metric_n - file_1.csv file_2.csv ...
"""
import csv
import sys

if __name__ == "__main__":
    i = 1
    metrics = []
    paths = []
    while i < len(sys.argv) and sys.argv[i] != '-':
        metrics.append(sys.argv[i])
        i += 1
    assert metrics, "No metrics given"

    if i < len(sys.argv) and sys.argv[i] == '-':
        i += 1
    while i < len(sys.argv):
        paths.append(sys.argv[i])
        i += 1
    assert paths, "No paths given"

    met_dict = {m:[] for m in metrics}
    for p in paths:
        with open(p, 'r') as r:
            reader = csv.DictReader(r) 
            row = next(reader)
            for m in metrics:
                met_dict[m].append(float(row[m]))

    met_means = {m:sum(entries)/len(entries) for m,entries in met_dict.items()}
    out_writer = csv.DictWriter(sys.stdout, fieldnames=metrics)
    out_writer.writeheader()
    out_writer.writerow(met_means)