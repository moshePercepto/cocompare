from helpers import write_csv, read_jsons, extract_cocos, get_analysis
FIELDS_NAMES = ['rgb_name', 'thermal_name', 'GT', 'PREDICT', 'FN', 'FP', 'TP']


class BaseModel:
    def __init__(self, name, th=0):
        self.name = name
        self.images: list = []
        self.images_list: list[str] = []
        self.images_count = 0
        self.pred_list = []
        self.gt_list = []
        self.gt_count = 0  # bboxs count for all GT test | known as group A
        self.pred_count = 0  # bboxs count for all prediction test | known as group B
        self.annotations = None
        self.th = th
        self.tp = self.fp = self.fn = 0

    def load_gt_data(self, paths):
        if type(paths) is str:  # verify if paths is only one path, to be in a list.
            paths = [paths]
        datas = read_jsons(paths)
        # print(f"{self.name}\n")
        self.gt_list += extract_cocos(datas)

    def get_info(self):
        return self.images_count, self.gt_count, self.pred_count, self.tp, self.fp, self.fn, self.th

    def get_analysis(self):
        recall = round(self.tp / self.gt_count, 2) if self.gt_count else 0
        precision = round(self.tp / self.pred_count, 2) if self.pred_count else 0
        f1_score = round((2 * self.tp) / (self.gt_count + self.pred_count), 2) if precision + recall else 0
        return recall, precision, f1_score, self.th

    def create_images(self, model):
        self.images += [model(self, data) for data in self.annotations]
        self.images.sort(key=lambda x: x.file_name)

    def __repr__(self):
        images_count, gt, pred, tp, fp, fn, th = self.get_info()
        recall, precision, f1_score = get_analysis(gt, pred, tp)
        return (f"{'-' * 100}\n{self.name=}\n{gt=}, {pred=}, {tp=}, {fp=}, {fn=}\n"
                f"{th=}, {precision=}, {recall=}, {f1_score=}\n{'-' * 100}")

    def get_data(self):
        return [image.export_data() for image in self.images]

    def export_data(self, file_name: str):
        data = self.get_data()
        write_csv(file_name, FIELDS_NAMES, data)

    def print_data(self):
        [print(d) for d in self.get_data()]
