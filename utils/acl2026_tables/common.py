import csv

def round_percent(num):
    num = 100 * num
    return f"{num:.2f}"

def read_test_row(path):
    with open(path, 'r') as r:
        return next(csv.DictReader(r))