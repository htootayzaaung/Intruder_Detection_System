import cv2  # Make sure to import cv2
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

import os
import json
import numpy as np
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from detectron2.evaluation import COCOEvaluator

def get_intruder_dicts(img_dir, json_file_path):
    with open(json_file_path) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for img_ann in imgs_anns["images"]:
        record = {}
        
        filename = os.path.join(img_dir, img_ann["file_name"])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = img_ann["id"]
        record["height"] = height
        record["width"] = width

        annos = [anno for anno in imgs_anns["annotations"] if anno["image_id"] == img_ann["id"]]
        objs = []
        for anno in annos:
            obj = {
                "bbox": anno["bbox"],
                "bbox_mode": BoxMode.XYWH_ABS,
                "category_id": anno["category_id"] - 1,
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

# Correct the paths according to your project structure
for d in ["train", "valid"]:  # If you have a test set, replace or add "test" accordingly
    DatasetCatalog.register(f"intruder_{d}", lambda d=d: get_intruder_dicts(f"./{d}", f"./annotations/{d}_annotations.json"))
    MetadataCatalog.get(f"intruder_{d}").set(thing_classes=["Axe", "Handgun", "Knife"])
intruder_metadata = MetadataCatalog.get("intruder_train")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("intruder_train",)
cfg.DATASETS.TEST = ("intruder_valid",)  # Adjust if you're using a different naming convention or dataset
cfg.DATALOADER.NUM_WORKERS = 4
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 2
cfg.SOLVER.BASE_LR = 0.0025
cfg.SOLVER.MAX_ITER = 3000
cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 3  # Number of classes
cfg.MODEL.DEVICE = 'cpu'
cfg.OUTPUT_DIR = "./output"

# Setup the evaluator
cfg.TEST.EVAL_PERIOD = 500
evaluator = COCOEvaluator("intruder_valid", cfg, False, output_dir=cfg.OUTPUT_DIR)

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = Trainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

# Model and config are saved in cfg.OUTPUT_DIR
print("Model and config saved to: ", cfg.OUTPUT_DIR)
