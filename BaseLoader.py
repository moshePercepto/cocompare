FIELDS_NAMES = ['group_name', 'GT', 'PREDICT', 'RECALL', 'F1_SCORE', 'FN', 'FP', 'TP']


class BaseLoader:
    def __init__(self, **options):
        self.model = options['model']
        self.gt = options['gt']
        self.pred = options['pred']
        self.th = options['th']
        self.d_th = options['d_th'] if self.model == 'ogi' else None
        self._index = 0
        self.children = []
        self.children_count = None
        self.gt_count = 0
        self.pred_count = 0
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def collect_children_data(self):
        pass

    def export_data(self, path: str):
        # FIELDS_NAMES = ['group_name', 'GT', 'PREDICT', 'RECALL', 'F1_SCORE', 'FN', 'FP', 'TP']
        return [
            {'group_name': child.name,
             'GT': child.gt_count,
             'PREDICT': child.pred_count,
             'TP': child.tp,
             'FP': child.fp,
             'FN': child.fn
             }
            for child in self.children]


    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, i):
        self._index = i

    @property
    def current(self):
        return self.children[self._index]

    @property
    def children_list(self):
        return [child.name for child in self.children]
