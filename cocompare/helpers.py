import json
import csv


def read_csv(path: str) -> list:
    with open(path, mode='r') as f:
        data = csv.DictReader(f)
        return [row for row in data]


def write_csv(path: str, field_names: list, data: list[dict]) -> None:
    with open(path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=field_names)
        writer.writeheader()
        writer.writerows(data)


def read_json(path: str) -> dict:
    with open(path, mode='r') as f:
        return json.load(f)


def read_jsons(path: str) -> list[dict]:
    return [read_json(p) for p in path]