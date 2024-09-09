import argparse
from helpers import *
from BaseLoader import BaseLoader


CRITERIA_FLAG = True
IOA = 'ioa'
FIELDS_NAMES = ['rgb_name', 'thermal_name', 'GT', 'PREDICT', 'FN', 'FP', 'TP']


class OgiLoader(BaseLoader):
    def build(self, ):
        gt_scenes = self.load_scenes_dir()
        pred_data = self.get_pred_data()
        self.children = self.build_scenes(gt_scenes, pred_data)
        self.children_count = len(self.children)

    def build_scenes(self, gt_scenes, pred_data):
        def get_coco_path(scene_name):
            for file_name in os.listdir(str(os.path.join(self.gt, scene_name))):
                if os.path.splitext(file_name)[-1] in ('.json', '.JSON'):
                    return os.path.join(self.gt, scene_name, file_name)

        scenes_names = sorted(list(set(gt_scenes).intersection(set(pred_data.keys()))))
        return [Ogi(
            name=scene_name,
            gt_path=get_coco_path(scene_name),
            pred_data=pred_data[scene_name],
            th=self.th
        ) for scene_name in scenes_names]

    def load_scenes_dir(self):
        """load all scene paths to a list"""
        return [folder for folder in os.listdir(self.gt) if os.path.isdir(os.path.join(self.gt, folder))]

    def get_pred_data(self):
        pred_scenes_data = read_json(self.pred)
        return {re_scene_name(name): data for name, data in pred_scenes_data['scenes'].items()}

    def collect_children_data(self):
        """
        ogi guidelines:
            if {self.d_th} of the frames in a scene is tp all the scene, We define a Hit,tp!
            else there is no tp for scene:
                if tp was found less than d_th, We define a Miss,fn!
                else if there is no gt annotation but there is pred annotation for all scene add 1 to global fp
        :return:
        """
        self.gt_count = self.pred_count = 0
        for child in self.children:
            images_count, gt_count, pred_count, tp, fp, fn, th = child.get_info()
            # print(f"{child.name} has: {images_count} frames")
            self.gt_count += gt_count if gt_count else 0
            self.pred_count += pred_count if pred_count else 0
            if get_precision(tp, gt_count) >= self.d_th:
                self.tp += 1
            else:
                self.fn += 1
            if get_precision(fp, images_count) >= self.d_th:
                self.fp += 1
            self.th = th
        return self.gt_count, self.pred_count, self.tp, self.fp, self.fn, self.th

    def gui_export(self):
        frames = ['scene_name', 'frame_name', 'gt', 'pred', 'git', 'false', 'miss', 'recall', 'precision', 'f1']
        scenes = ['scene_name', 'gt', 'pred', 'git', 'false', 'miss', 'recall', 'precision', 'f1']
        total = ['gt', 'pred', 'git', 'false', 'miss', 'recall', 'precision', 'f1']
        a = {'frames': [frame.gui_info(child.name) for child in self.children for frame in child.images],
             'scenes': [child.gui_info() for child in self.children],
             'total': self.gui_info(),
             }

    def gui_info(self):
        gt_count, pred_count, tp, fp, fn, _ = self.collect_children_data()
        return {'gt': gt_count, 'pred': pred_count, 'hit': tp, 'false': fp, 'miss': fn,
                'precision': get_precision(tp, fp),
                'recall': get_recall(tp, fn),
                'f1': get_f1_score(get_recall(tp, fn), get_precision(tp, fp))
                }

    def __repr__(self):
        gt_count, pred_count, tp, fp, fn, th = self.collect_children_data()
        recall, precision, f1_score = get_analysis(tp, fp, fn)
        return (f"{'-' * 100}\nOGI Summery:\n{gt_count=}, {pred_count=}, {tp=}, {fp=}, {fn=}\n"
                f"{th=}, {precision=}, {recall=}, {f1_score=}\n{'-' * 100}")


class Ogi:
    def __init__(self, name, gt_path, pred_data, th):
        self.name = name
        print(self.name)
        self.gt_list = self.load_gt_data(gt_path)
        self.pred_list = extract_coco(pred_data, self.get_pred_bboxes)
        # fill the below with process_data method
        self.names_list = self.get_images_names()
        self.images_count = len(self.names_list)
        self.gt_count = 0
        self.pred_count = 0
        self.annotations = self.build_annotations()
        self.th = th
        self.tp = self.fp = self.fn = 0
        self.images: list[OgiImage] = self.create_images()

    def load_pred_data(self, path):
        if os.path.split(path)[1].lower() == 'csv':
            data = read_csv(path)
            # return self.extract_pred_from_csv(data)
        else:
            data = read_json(path)
            return self.extract_pred_from_json(data)

    def extract_pred_from_json(self, data):
        return [{
            'thermal_name': d['thermal_name'],
            'rgb_name': d['rgb_name'],
            'width': d['thermal_shape'][1],
            'height': d['thermal_shape'][0],
            'bboxes': d['bboxes']
        } for d in data]

    def load_gt_data(self, paths):
        if type(paths) is str:  # verify if paths is only one path, to be in a list.
            paths = [paths]
        cocos = read_jsons(paths)
        return extract_cocos(cocos, self.get_gt_bboxes)

    def extract_cocos(self, cocos, bboxes_callback):
        return [{
            'file_name': image['file_name'],
            'width': image['width'],
            'height': image['height'],
            'bboxes': bboxes_callback(coco, image)
        } for coco in cocos for image in coco['images']]

    def get_gt_bboxes(self, coco, image):
        return [a['bbox'] for a in coco['annotations'] if a['image_id'] == image['id']]

    def get_pred_bboxes(self, coco, image):
        return [[a['bbox'], a['confidence']]for a in coco['annotations'] if a['image_id'] == image['id']]

    def get_images_names(self, ):
        set1 = {pred_ann['file_name'] for pred_ann in self.pred_list}
        set2 = {gt_ann['file_name'] for gt_ann in self.gt_list}
        return sorted(list(set1.intersection(set2)))

    def get_info(self):
        return self.images_count, self.gt_count, self.pred_count, self.tp, self.fp, self.fn, self.th

    def create_images(self):
        images = [OgiImage(self, data) for data in self.annotations]
        images.sort(key=lambda x: x.name)
        return images

    def build_annotations(self):
        annotations = []
        for file_name in self.names_list:
            new_ann = {'thermal_name': file_name, 'gt_bboxes': []}
            for pred_ann in self.pred_list:
                if file_name == pred_ann['file_name']:
                    new_ann.update({
                        'width': pred_ann['width'],
                        'height': pred_ann['height'],
                        'pred_bboxes': pred_ann['bboxes']
                    })
                    self.pred_count += len(pred_ann['bboxes'])
            for gt_ann in self.gt_list:
                if file_name == gt_ann['file_name']:
                    new_ann['gt_bboxes'] += (gt_ann['bboxes'])
                    self.gt_count += len(gt_ann['bboxes'])
            annotations.append(new_ann)
        # print('\n', self.name, '\n', annotations)
        return annotations

    def get_image_index(self, file_name: str) -> int:
        for i in range(len(self.images)):
            if self.images[i].name == file_name:
                return i

    def get_data(self):
        return [image.export_data() for image in self.images]

    def export_data(self, file_name: str):
        data = self.get_data()
        write_csv(file_name, FIELDS_NAMES, data)

    def print_data(self):
        [print(d) for d in self.get_data()]

    def __repr__(self):
        scene_name = self.name
        images_count, gt, pred, tp, fp, fn, th = self.get_info()
        recall, precision, f1_score = get_analysis(tp, fp, fn)
        return (f"{'-' * 100}\n{scene_name=}\n{gt=}, {pred=}, {tp=}, {fp=}, {fn=}\n"
                f"{th=}, {precision=}, {recall=}, {f1_score=}\n{'-' * 100}")

    def get_analysis(self):
        _, _, _, tp, fp, fn, th = self.get_info()
        recall, precision, f1_score = get_analysis(tp, fp, fn)
        return recall, precision, f1_score, th

    def gui_info(self):
        return {'scene_name': self.name,
                'gt': self.gt_count,
                'pred': self.pred_count,
                'hit': self.tp,
                'false': self.fp,
                'miss': self.fn,
                'precision': get_precision(self.tp, self.fp),
                'recall': get_recall(self.tp, self.fn),
                'f1': get_f1_score(get_recall(self.tp, self.fn), get_precision(self.tp, self.fp))
                }


class OgiImage:
    def __init__(self, manager, data):
        self.manager: Ogi = manager
        self.name = data['thermal_name']
        self.width = data['width']
        self.height = data['height']
        self.pred, self.confidence = self.get_bboxes_and_confidence_from_pred(data['pred_bboxes'])
        print(self.name)
        print(f"{self.pred}: {self.confidence=}")
        self.gt = self.normalize_bboxes(data['gt_bboxes'])
        self.metric = []
        self.tp = self.fp = self.fn = 0
        self.check_collisions()
        self.count_metrics()

    def export_data(self):
        return {
            'scene_name': self.manager.name,
            'file_name': self.name,
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
        for row in self.metric:
            self.tp += 1 if sum(row) else 0
            self.manager.tp += 1 if sum(row) else 0
            if sum(row) == 0:
                self.fn += 1
                self.manager.fn += 1
        # Count False Positives (FP)
        if self.gt:
            for col_idx in range(len(self.metric[0])):
                column_sum = sum([self.metric[row_idx][col_idx] for row_idx in range(len(self.metric))])
                self.fp += 0 if column_sum else 1
                self.manager.fp += 0 if column_sum else 1
        elif self.pred:
            self.fp += 1
            self.manager.fp += 1

    def check_collisions(self):
        for gt_bbox in self.gt:
            gt_bbox = OgiImage.bbox2pt(gt_bbox)

            row = [self.calc_iou(gt_bbox, OgiImage.bbox2pt(pred_bbox), self.manager.th) for pred_bbox in self.pred]
            self.metric.append(row)

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

    def gui_info(self, scene_name):
        return {
            'scene_name': scene_name,
            'frame_name': self.name,
            'gt': len(self.gt),
            'pred': len(self.pred),
            'hit': self.tp,
            'false': self.fp,
            'miss': self.fn,
            'recall': self.fn,
            'precision': self.fn,
            'f1': self.fn,
        }

    def get_bboxes_and_confidence_from_pred(self, annotations):
        bboxes_list = []
        confidence_list = []
        for bbox, confidence in annotations:
            bboxes_list.append(self.normalize_bbox(bbox, self.width, self.height))
            confidence_list.append(confidence)
        return bboxes_list, confidence_list


def parse_arguments():
    parser = argparse.ArgumentParser(description="Handle file paths and options")
    parser.add_argument(
        '-p', '--pred',
        # dest='pred',
        type=str,
        # nargs='+',  # This allows multiple strings to be provided
        required=True,
        help="List of predictions paths (mandatory)"
    )
    parser.add_argument(
        '-g', '--gt',
        type=str,
        # nargs='+',  # This allows multiple strings to be provided
        required=True,
        help="A gt path (mandatory)"
    )
    parser.add_argument(
        '-th', '-th',
        type=float,
        required=True,
        dest='th',
        help="thresh-hold condition for iou"
    )
    parser.add_argument(
        '-dth', '-dth',
        type=float,
        required=True,
        dest='dth',
        help="detect-thresh-hold condition for general calculations"
    )
    parser.add_argument(
        '-s', '--save',
        type=str,
        default=None,  # This makes it optional
        help="An optional string"
    )
    return parser.parse_args()


def ogi(args):
    opt = {'model': 'ogi', 'gt': args.gt, 'pred': args.pred, 'th': args.th, 'd_th': args.dth}

    ogi_loader = OgiLoader(**opt)
    # print(ogi_loader)
    ogi_loader.build()
    # for t in ogi_loader.children:
    #     print(t)
    print(ogi_loader)
    # print(ogi_loader.collect_children_data())
    # if args.save:
    #     ogi_loader.export_data(args.save)


if __name__ == '__main__':
    args = parse_arguments()
    # pred = (r'C:\Users\MosheMendelovich\Documents\percepto\cocompare\data\ogi'
    #         r'\predictions_Leaks_dataset_V5_13_prod_2024-09-04-12-46-08.json')
    # args.pred = pred
    ogi(args)
