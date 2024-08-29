import os
from helpers import read_json, re_scene_name, extract_coco, get_analysis, get_precision
from base_model import BaseModel
from image_model import OgiImage


class OgiLoader:
    def __init__(self, scenes_path: str, pred_path: str, th: float = 0.3):
        self.scenes_names = None
        self.scenes = OgiScenes()
        self.scenes_length = 0
        self.scenes_path = scenes_path
        self.pred_path = pred_path
        self.th = th
        self.load_scenes()

    def load_scenes(self):
        gt_scenes = self.load_scenes_dir()
        print(f"{len(gt_scenes)=} scenes loaded")
        pred_data = self.get_pred_data()
        self.build_scenes(gt_scenes, pred_data)

    def load_scenes_dir(self):
        return [f for f in os.listdir(self.scenes_path) if os.path.isdir(os.path.join(self.scenes_path, f))]

    def get_pred_data(self):
        pred_scenes_data = read_json(self.pred_path)
        return {re_scene_name(name): data for name, data in pred_scenes_data['scenes'].items()}

    def build_scenes(self, gt_scenes, pred_data):
        def get_coco_path(scene_name):
            for file_name in os.listdir(os.path.join(self.scenes_path, scene_name)):
                if os.path.splitext(file_name)[-1] in ('.json', '.JSON'):
                    return os.path.join(self.scenes_path, scene_name, file_name)

        self.scenes_names = list(set(gt_scenes).intersection(set(pred_data.keys())))
        [self.scenes.add_scene(OgiModel(
            name=scene_name,
            gt_path=get_coco_path(scene_name),
            pred_data=pred_data[scene_name],
            th=self.th
        )) for scene_name in self.scenes_names]


class OgiModel(BaseModel):
    def __init__(self, name, gt_path, pred_data, th):
        super().__init__(name, th)

        self.load_gt_data(gt_path)
        self.pred_list = extract_coco(pred_data)
        self.get_images_names()
        self.annotations = self.build_annotations()
        self.create_images(OgiImage)
        # print(f"{name=}\n{self.images_list=}\n{pred_data.keys()=}\n{self.pred_list=}\n\n")

    def get_images_names(self, ):
        set1 = {pred_ann['file_name'] for pred_ann in self.pred_list}
        set2 = {gt_ann['file_name'] for gt_ann in self.gt_list}
        self.images_list = sorted(list(set1.intersection(set2)))
        self.images_count = len(self.images_list)

    def build_annotations(self):
        annotations = []
        print(self.name)
        for file_name in self.images_list:
            new_ann = {'file_name': file_name}
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
                    new_ann.update({'gt_bboxes': gt_ann['bboxes']})
                    self.gt_count += len(gt_ann['bboxes'])
            annotations.append(new_ann)
        # print('\n', self.name, '\n', annotations)
        return annotations


class OgiScenes:
    def __init__(self, d_th=0.02):
        self.scenes = []
        self.d_th = d_th  # Detection threshold
        self.gt_count = self.pred_count = self.tp = self.fp = self.fn = self.th = 0

    def add_scene(self, ogi_scene: OgiModel):
        self.scenes.append(ogi_scene)

    # thief_hatch_scene_1_120052 000110
    def collect_scenes_data(self):
        """
        ogi guidelines:
            if {self.d_th} of the frames in a scene is tp all the scene, We define a Hit,tp!
            else there is no tp for scene:
                if tp was found less than d_th, We define a Miss,fn!
                else if there is no gt annotation but there is pred annotation for all scene add 1 to global fp
        :return:
        """
        for ogi_scene in self.scenes:
            images_count, gt_count, pred_count, tp, fp, fn, th = ogi_scene.get_info()
            print(f"{ogi_scene.name} has: {images_count} frames")
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

    def __repr__(self):
        gt_count, pred_count, tp, fp, fn, th = self.collect_scenes_data()
        recall, precision, f1_score = get_analysis(gt_count, pred_count, tp)
        return (f"{'-' * 100}\nogi summery:\n{gt_count=}, {pred_count=}, {tp=}, {fp=}, {fn=}\n"
                f"{th=}, {precision=}, {recall=}, {f1_score=}\n{'-' * 100}")


if __name__ == '__main__':
    gt = r'C:\Users\MosheMendelovich\Documents\percepto\cocompare\data\ogi\gt'
    pred = r'C:\Users\MosheMendelovich\Documents\percepto\cocompare\data\ogi\predictions_2024-08-28-10-30-41.json'
    ogi = OgiLoader(gt, pred)
    ogi_scenes = ogi.scenes.scenes
    for scene in ogi_scenes:
        print(scene.print_data())
        print(scene)
    print(ogi.scenes)
