import json
import csv
import re


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


def re_scene_name(name):
    pattern = r'^[^_]+_[^_]+_(.+)_[^_]+$'
    match = re.match(pattern, name)
    if match:
        return match.group(1)
    else:
        return ''


def extract_cocos(datas):
    # for data in datas:
    #
    #     for a in data['annotations']:
    #         if a['bbox']:
    #             print(a['image_id'])

    return [{
        'file_name': image['file_name'],
        'width': image['width'],
        'height': image['height'],
        'bboxes': [a['bbox'] for a in data['annotations'] if a['image_id'] == image['id']]
    } for data in datas for image in data['images']]


def extract_coco(data):
    return [{
        'file_name': image['file_name'],
        'width': image['width'],
        'height': image['height'],
        'bboxes': [a['bbox'] for a in data['annotations'] if a['image_id'] == image['id']]
    } for image in data['images']]


# def extract_multi_cocos(cocos):
#     data = []
#     data += [extract_coco(coco) for coco in cocos]
#     print(len(data))
#
#
# def extract_coco(coco):
#     return [{
#         'file_name': image['file_name'],
#         'width': image['width'],
#         'height': image['height'],
#         'bboxes': [a['bbox'] for a in coco['annotations'] if a['image_id'] == image['id']]
#     } for image in coco['images']]

def get_recall(tp, gt_length):
    return round(tp / gt_length, 2) if gt_length else 0


def get_precision(tp, pred_length):
    return round(tp / pred_length, 2) if pred_length else 0


def get_analysis(gt_length, pred_length, tp):
    recall = get_recall(tp, gt_length)
    precision = get_precision(tp, pred_length)
    f1_score = round((2 * tp) / (gt_length + pred_length), 2) if precision + recall else 0
    return recall, precision, f1_score
