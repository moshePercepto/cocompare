from base_model import BaseModel


class BaseImageModel:
    def __init__(self, manager, data):
        self.manager: BaseModel = manager
        self.file_name = data['thermal_name'] if 'thermal_name' in data else data['file_name']
        self.width = data['width']
        self.height = data['height']
        self.pred = self.normalize_bboxes(data['pred_bboxes'])
        self.gt = self.normalize_bboxes(data['gt_bboxes'])
        self.metric = []
        self.tp = self.fp = self.fn = 0
        self.check_collisions()

    def count_metrics(self):
        for row in self.metric:
            self.tp += 1 if sum(row) else 0
            self.manager.tp += 1 if sum(row) else 0
            if sum(row) == 0:
                self.fn += 1
                self.manager.fn += 1
        # Count False Positives (FP)
        # num_columns = len(self.metric[0]) if self.metric[0] else 0
        if self.gt:
            for col_idx in range(len(self.metric[0])):
                column_sum = sum([self.metric[row_idx][col_idx] for row_idx in range(len(self.metric))])
                self.fp += 0 if column_sum else 1
                self.manager.fp += 0 if column_sum else 1
        elif self.pred:
            self.fp += 1
            self.manager.fp += 1

    def print_metric(self):
        print(f"\nmetric:")
        for row in self.metric:
            print(f"{row}")
        print(f"end of metric:\n")

    def get_info(self):
        return len(self.gt), len(self.pred), self.tp, self.fp, self.fn

    def normalize_bboxes(self, bboxes):
        return [self.normalize_bbox(bbox, self.width, self.height) for bbox in bboxes]

    @staticmethod
    def normalize_bbox(bbox, width, height):
        x, y, w, h = bbox
        return [x / width, y / height, w / width, h / height]

    def check_collisions(self):
        for gt_bbox in self.gt:
            gt_bbox = TndImage.bbox2pt(gt_bbox)
            row = [self.calc_iou(gt_bbox, TndImage.bbox2pt(pred_bbox), self.manager.th) for pred_bbox in self.pred]
            self.metric.append(row)


    @staticmethod
    def calc_iou(gt_bbox: list, pred_bbox: list, th: float):
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
        return iou if iou >= th else 0
        # return ioa if ioa >= condition else 0

    @staticmethod
    def bbox2pt(bbox) -> list:
        return [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

    def __repr__(self):
        return (f"\n{'-' * 100}\n{self.file_name}\nwidth: {self.width} height: "
                f"{self.height}\n{self.pred}\n{self.gt}\n{self.metric}\n{'-' * 100}\n")


class TndImage(BaseImageModel):
    def __init__(self, manager, data):
        super().__init__(manager, data)
        self.rgb_name = data['rgb_name']
        self.count_metrics()

    def export_data(self):
        return {
            'rgb_name': self.rgb_name,
            'thermal_name': self.file_name,
            'GT': len(self.gt),
            'PREDICT': len(self.pred),
            'TP': self.tp,
            'FP': self.fp,
            'FN': self.fn
        }


class OgiImage(BaseImageModel):
    def __init__(self, manager, data):
        super().__init__(manager, data)
        self.count_metrics()

    def export_data(self):
        return {
            'scene_name': self.manager.name,
            'file_name': self.file_name,
            'GT': len(self.gt),
            'PREDICT': len(self.pred),
            'TP': self.tp,
            'FP': self.fp,
            'FN': self.fn
        }
