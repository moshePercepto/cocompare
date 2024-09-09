import json
import csv
import re
import os


def is_csv(path: str) -> bool:
    return os.path.splitext(path)[1].lower() == '.csv'


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


def extract_pred_json(data):
    return [{
        'thermal_name': d['thermal_name'],
        'rgb_name': d['rgb_name'],
        'width': d['thermal_shape'][1],
        'height': d['thermal_shape'][0],
        'bboxes': d['bboxes']
    } for d in data]


def extract_gt_csv(datas):
    def extract_file_name(filename):
        return '__'.join(filename.split('__')[3:-1]) + '.jpg'

    def get_gt_bbox(bboxs):
        return [[x1, y1, x2 - x1, y2 - y1] for bbox in bboxs for x1, y1, x2, y2 in [bbox]]

    return [{
        'file_name': extract_file_name(d['Thermal Image']),
        'bboxes': get_gt_bbox([[float(d['X_min']), float(d['Y_min']), float(d['X_max']), float(d['Y_max'])]])
    } for d in datas]


def extract_cocos(cocos, bboxes_callback=None):
    return [extract_coco(coco, bboxes_callback if bboxes_callback else get_gt_bboxes) for coco in cocos][0]


def extract_coco(coco, bboxes_callback):
    return [{
        'file_name': image['file_name'],
        'width': image['width'],
        'height': image['height'],
        'bboxes': bboxes_callback(coco, image)
    }for image in coco['images']]


def get_gt_bboxes(coco, image):
    return [a['bbox'] for a in coco['annotations'] if a['image_id'] == image['id']]


def get_pred_bboxes(coco, image):
    return [[a['bbox'], a['confidence']]for a in coco['annotations'] if a['image_id'] == image['id']]


def get_recall(tp, fn):
    return round(tp / (tp + fn), 2) if (tp + fn) else 0


def get_precision(tp, fp):
    return round(tp / (tp + fp), 2) if (tp + fp) else 0


def get_f1_score(recall, precision):
    return round((2 * recall * precision) / (recall + precision), 2) if precision + recall else 0


def get_analysis(tp, fp, fn):
    recall = get_recall(tp, fn)
    precision = get_precision(tp, fp)
    f1_score = get_f1_score(recall, precision)
    return recall, precision, f1_score
