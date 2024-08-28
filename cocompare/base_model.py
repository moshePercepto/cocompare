from image_model import Image


class BaseModel:
    def __init__(self, c_th):
        self.images: list[Image] = []
        self.images_list: list[str] = []
        self.pred_list = []
        self.gt_list = []
        self.gt_length = 0
        self.pred_length = 0
        self.annotations = None
        self.condition = c_th
        self.tp = self.fp = self.tn = 0

    def get_info(self):
        return self.gt_length, self.pred_length, self.tp, self.fp, self.tn

    def get_analysis(self):
        recall = round(self.tp / self.gt_length, 2) if self.gt_length else 0
        precision = round(self.tp / self.pred_length, 2) if self.pred_length else 0
        f1_score = round((2 * self.tp) / (self.gt_length + self.pred_length), 2) if precision + recall else 0
        return recall, precision, f1_score, self.condition
