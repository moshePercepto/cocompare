class Image:
    def __init__(self, manager, data):
        self.manager: Tnd = manager
        self.file_name = data['thermal_name']
        self.rgb_name = data['rgb_name']
        self.width = data['width']
        self.height = data['height']
        self.pred = self.normalize_bboxes(data['pred_bboxes'])
        self.gt = self.normalize_bboxes(data['gt_bboxes'])
        self.metric = []
        self.tp = self.fp = self.tn = 0
        self.check_collisions()