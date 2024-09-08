import argparse
import os
from helpers import *
from BaseLoader import BaseLoader


CRITERIA_FLAG = True
IOA = 'ioa'
FIELDS_NAMES = ['rgb_name', 'thermal_name', 'GT', 'PREDICT', 'FN', 'FP', 'TP']


class TndLoader(BaseLoader):
    def build(self):
        self.children = [Tnd(
            gt_paths=self.gt,
            pred_path=path,
            th=self.th
        ) for path in self.pred]

    def collect_children_data(self):
        for child in self.children:
            images_count, gt_count, pred_count, tp, fp, fn, th = child.get_info()
            self.gt_count += gt_count
            self.pred_count += pred_count
            self.tp += tp
            self.fp += fp
            self.fn += fn
            self.th = th
        return self.gt_count, self.pred_count, self.tp, self.fp, self.fn, self.th

    def __repr__(self):
        gt_count, pred_count, tp, fp, fn, th = self.collect_children_data()
        recall, precision, f1_score = get_analysis(gt_count, pred_count, tp)
        return (f"{'-' * 100}\ntnd summery:\n{gt_count=}, {pred_count=}, {tp=}, {fp=}, {fn=}\n"
                f"{th=}, {precision=}, {recall=}, {f1_score=}\n{'-' * 100}")


class Tnd:
    def __init__(self, gt_paths, pred_path, th):
        self.name = os.path.basename(pred_path).split('.')[0]
        self.gt_list = self.load_gt_data(gt_paths)
        self.pred_list = self.load_pred_data(pred_path)
        # fill the below with process_data method
        self.images_list = self.get_images_names()
        self.images_count = len(self.images_list)
        self.gt_count = 0
        self.pred_count = 0
        self.annotations = self.build_annotations()
        self.th = th
        self.tp = self.fp = self.fn = 0
        self.images: list[TndImage] = self.create_images()

    def load_pred_data(self, path):
        if is_csv(path):
            data = read_csv(path)
            # return self.extract_pred_from_csv(data)
        else:
            data = read_json(path)
            return extract_pred_json(data)

    def load_gt_data(self, paths):
        if type(paths) is str:  # verify if paths is only one path, to be in a list.
            paths = [paths]
        if is_csv(paths[0]):
            datas = read_csv(paths[0])
            return extract_gt_csv(datas)
        else:
            datas = read_jsons(paths)
            return extract_cocos(datas)

    def get_images_names(self, ):
        """intersection_of_lists"""
        set1 = {pred_ann['thermal_name'] for pred_ann in self.pred_list}
        set2 = {gt_ann['file_name'] for gt_ann in self.gt_list}
        return sorted(list(set1.intersection(set2)))

    def get_info(self):
        return self.images_count, self.gt_count, self.pred_count, self.tp, self.fp, self.fn, self.th

    def create_images(self):
        images = [TndImage(self, data) for data in self.annotations]
        images.sort(key=lambda x: x.name)
        return images

    def build_annotations(self):
        annotations = []
        for file_name in self.images_list:
            new_ann = {'thermal_name': file_name, 'gt_bboxes': []}
            for pred in self.pred_list:
                if file_name in pred['thermal_name']:
                    new_ann.update({
                        'rgb_name': pred['rgb_name'],
                        'width': pred['width'],
                        'height': pred['height'],
                        'pred_bboxes': pred['bboxes']
                    })
            for gt in self.gt_list:
                if file_name in gt['file_name']:
                    new_ann['gt_bboxes'] += (gt['bboxes'])
            self.pred_count += len(new_ann['pred_bboxes'])
            self.gt_count += len(new_ann['gt_bboxes'])
            annotations.append(new_ann)
        return annotations

    def get_image_index(self, file_name: str) -> int:
        for i, image in enumerate(self.images):
            if image.name == file_name:
                return i

    def get_data(self):
        return [image.export_data() for image in self.images]

    def export_data(self, path: str):
        data = self.get_data()
        write_csv(path, FIELDS_NAMES, data)

    def print_data(self):
        [print(d) for d in self.get_data()]

    def __repr__(self):
        name = self.name
        images_count, gt, pred, tp, fp, fn, th = self.get_info()
        recall, precision, f1_score = get_analysis(gt, pred, tp)
        return (f"{'-' * 50}\n{name=}\n{gt=}, {pred=}, {tp=}, {fp=}, {fn=}\n"
                f"{th=}, {precision=}, {recall=}, {f1_score=}\n{'-' * 50}")


class TndImage:
    def __init__(self, manager, data):
        self.manager: Tnd = manager
        self.name = data['thermal_name']
        self.rgb_name = data['rgb_name']
        self.width = data['width']
        self.height = data['height']
        self.pred = self.normalize_bboxes(data['pred_bboxes'])
        self.gt = self.normalize_bboxes(data['gt_bboxes'])
        self.metric = []
        self.tp = self.fp = self.fn = 0
        self.check_collisions()

    def export_data(self):
        return {
            'rgb_name': self.rgb_name,
            'thermal_name': self.name,
            'GT': len(self.gt),
            'PREDICT': len(self.pred),
            'TP': self.tp,
            'FP': self.fp,
            'FN': self.fn
        }

    def get_info(self):
        return len(self.gt), len(self.pred), self.tp, self.fp, self.fn

    def normalize_bboxes(self, bboxes):
        return [self.normalize_bbox(bbox, self.width, self.height) for bbox in bboxes]

    @staticmethod
    def normalize_bbox(bbox, width, height):
        x, y, w, h = bbox
        return [x / width, y / height, w / width, h / height]

    def count_metrics(self):
        # self.print_metric()
        for row in self.metric:
            self.tp += 1 if sum(row) else 0
            self.manager.tp += 1 if sum(row) else 0
            if sum(row) == 0:
                self.fn += 1
                self.manager.fn += 1
        # Count False Positives (FP)
        num_columns = len(self.metric[0])
        for col_idx in range(num_columns):
            column_sum = sum([self.metric[row_idx][col_idx] for row_idx in range(len(self.metric))])
            self.fp += 0 if column_sum else 1
            self.manager.fp += 0 if column_sum else 1
        # print(f"True Positives (TP): {self.tp}")
        # print(f"True Negatives (TN): {self.tn}")
        # print(f"False Positives (FP): {self.fp}")

    def check_collisions(self):
        for gt_bbox in self.gt:
            gt_bbox = TndImage.bbox2pt(gt_bbox)

            row = [self.calc_iou(gt_bbox, TndImage.bbox2pt(pred_bbox), self.manager.th) for pred_bbox in self.pred]
            self.metric.append(row)
        # self.count_metrics()

    def print_metric(self):
        print(f"\nmetric:")
        for row in self.metric:
            print(f"{row}")
        print(f"end of metric:\n")

    @staticmethod
    def bbox2pt(bbox) -> list:
        return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

    @staticmethod
    def calc_iou(gt_bbox: list, pred_bbox: list, condition: float):
        x1_a, y1_a, x2_a, y2_a = gt_bbox
        x1_b, y1_b, x2_b, y2_b = pred_bbox
        # Calculate the area of intersection
        inter_area = max(0, min(x2_a, x2_b) - max(x1_a, x1_b)) * max(0, min(y2_a, y2_b) - max(y1_a, y1_b))
        # Calculate the areas of each rectangle
        pred_area = (x2_b - x1_b) * (y2_b - y1_b)
        gt_area = (x2_a - x1_a) * (y2_a - y1_a)
        # Compute the IoU
        union_area = pred_area + gt_area - inter_area
        iou = inter_area / union_area if union_area != 0 else 0
        # Compute the IoA
        # ioa = inter_area / pred_area if pred_area != 0 else 0
        return iou if iou >= condition else 0
        # return ioa if ioa >= condition else 0

    def __repr__(self):
        return (f"\n{'-' * 100}\n{self.name}\nwidth: {self.width} height: "
                f"{self.height}\n{self.pred}\n{self.gt}\n{self.metric}\n{'-' * 100}\n")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Handle file paths and options")
    parser.add_argument(
        '-p', '--pred',
        type=str,
        nargs='+',  # This allows multiple strings to be provided
        required=True,
        help="List of predictions paths (mandatory)"
    )
    parser.add_argument(
        '-g', '--gt',
        type=str,
        nargs='+',  # This allows multiple strings to be provided
        required=True,
        help="A gt path (mandatory)"
    )
    parser.add_argument(
        '-th', '--th',
        type=float,
        required=True,
        help="thresh_hold condition for iou"
    )
    parser.add_argument(
        '-s', '--save',
        type=str,
        default=None,  # This makes it optional
        help="An optional string"
    )
    return parser.parse_args()


if __name__ == '__main__':
    def tnd_ind(args):
        tnd_loader = TndLoader(model='tnd', gt=args.gt, pred=args.pred, th=args.th)
        tnd_loader.build()
        for child in tnd_loader.children:
            print(f"\n{child}")
            if args.save:
                save_path = f"{os.path.splitext(args.save)[0]}_{child.name}.csv"
                child.export_data(save_path)

    def tnd_gen(args):
        tnd_loader = TndLoader(model='tnd', gt=args.gt, pred=args.pred, th=args.th)
        tnd_loader.build()

        if args.save:
            tnd_loader.export_data(args.save)
        else:
            # tnd.print_data()
            print(tnd_loader)

    args = parse_arguments()
    tnd_ind(args)
    tnd_gen(args)
