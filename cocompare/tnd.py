import os

from overrides import overrides

from image_model import TndImage
from helpers import read_json, get_analysis
from base_model import BaseModel


class TndLoader:
    def __init__(self, gt_paths: list[str], pred_paths: list[str], th: float = 0.3):
        self.tnd_tests = TndTests()
        self.tests_length = 0
        self.gt_paths = gt_paths
        self.pred_paths = pred_paths
        self.th = th
        self.build_tests()

    def build_tests(self,):
        def get_test_name(path):
            return os.path.basename(path).split('.')[0]

        # get_test_name(self.pred_paths[0])
        [self.tnd_tests.add_tests(TndModel(
            name=get_test_name(path),
            gt_paths=self.gt_paths,
            pred_path=path,
            th=self.th
        )) for path in self.pred_paths]


class TndModel(BaseModel):
    def __init__(self, name, gt_paths, pred_path, th):
        super().__init__(name, th)
        self.load_pred_data(pred_path)
        self.load_gt_data(gt_paths)
        self.get_images_names()
        self.annotations = self.build_annotations()
        self.create_images(TndImage)

    def load_pred_data(self, path: str):
        def extract_pred_from_json(data):
            return [{
                'thermal_name': d['thermal_name'],
                'rgb_name': d['rgb_name'],
                'width': d['thermal_shape'][1],
                'height': d['thermal_shape'][0],
                'bboxes': d['bboxes']
            } for d in data]

        self.pred_list += extract_pred_from_json(read_json(path))

    def get_images_names(self,):
        set1 = {pred_ann['thermal_name'] for pred_ann in self.pred_list}
        set2 = {gt_ann['file_name'] for gt_ann in self.gt_list}
        self.images_list = sorted(list(set1.intersection(set2)))
        self.images_count = len(self.images_list)

    def process_data(self):
        self.annotations = self.build_annotations()
        self.create_images(TndImage)

    def build_annotations(self):
        annotations = []
        for file_name in self.images_list:
            new_ann = {'thermal_name': file_name}
            for pred in self.pred_list:
                if file_name == pred['thermal_name']:
                    new_ann.update({
                        'rgb_name': pred['rgb_name'],
                        'width': pred['width'],
                        'height': pred['height'],
                        'pred_bboxes': pred['bboxes']
                    })
                    self.pred_count += len(pred['bboxes'])
            for gt in self.gt_list:
                if file_name == gt['file_name']:
                    new_ann.update({'gt_bboxes': gt['bboxes']})
                    self.gt_count += len(gt['bboxes'])
            annotations.append(new_ann)
        return annotations


class TndTests:
    def __init__(self):
        self.tests = []
        self.gt_count = self.pred_count = self.tp = self.fp = self.fn = self.th = 0

    def add_tests(self, tnd_model: TndModel):
        self.tests.append(tnd_model)

    def collect_test_data(self):
        for tnd_model in self.tests:
            images_count, gt_count, pred_count, tp, fp, fn, th = tnd_model.get_info()
            self.gt_count += gt_count
            self.pred_count += pred_count
            self.tp += tp
            self.fp += fp
            self.fn += fn
            self.th = th
        return self.gt_count, self.pred_count, self.tp, self.fp, self.fn, self.th

    def __repr__(self):
        gt_count, pred_count, tp, fp, fn, th = self.collect_test_data()
        recall, precision, f1_score = get_analysis(gt_count, pred_count, tp)
        return (f"{'-' * 100}\ntnd summery:\n{gt_count=}, {pred_count=}, {tp=}, {fp=}, {fn=}\n"
                f"{th=}, {precision=}, {recall=}, {f1_score=}\n{'-' * 100}")


if __name__ == '__main__':
    gt = [r'C:\Users\MosheMendelovich\Documents\percepto\cocompare\data\tnd\COCO_test_R2T_oldata_BB_issues.json']
    pred = [
        # r"C:\Users\MosheMendelovich\PycharmProjects\data_scrapping\M3T_anomalies_meregd_02.json",
        # r"C:\Users\MosheMendelovich\PycharmProjects\data_scrapping\M30T_anomalies_meregd_02.json",
        r"C:\Users\MosheMendelovich\PycharmProjects\data_scrapping\ZH20T_anomalies_meregd_02.json",
    ]
    tnd = TndLoader(gt, pred)
    tnd_tests = tnd.tnd_tests.tests
    for test in tnd_tests:
        print(test.print_data())
        print(test)
    print(tnd.tnd_tests)
