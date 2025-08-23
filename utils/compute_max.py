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

    out_rows = []
    for p in paths:
        p_mets = {m:0 for m in metrics}
        with open(p, 'r') as r:
            reader = csv.DictReader(r) 
            for row in reader:
                for m in filter(lambda x: row[x], metrics):
                    p_mets[m] = max(p_mets[m], float(row[m]))
        p_mets['path'] = p
        out_rows.append(p_mets)

    out_writer = csv.DictWriter(sys.stdout, fieldnames=['path'] + metrics)
    out_writer.writeheader()
    out_writer.writerows(out_rows)