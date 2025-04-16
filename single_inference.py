import torch
from loguru import logger
import config
from src.core import YAMLConfig
from utils.trainer import DetTrainer
import warnings
warnings.filterwarnings('ignore')

"""
this is the code for object detection
"""

logger.add('output/test/test_log.txt')
config.model_yaml = 'configs/rtdetr/rtdetr_r50vd_6x_coco_test.yml'
config.resume = 'result/rt50_nl50_niou_nobboxloss/best_model54.pth'
config.tuning = ''
config.test_only = True
config.device = 'cuda'

images_src = 'data/strands/'

if __name__ == '__main__':
    # 初始化
    cfg = YAMLConfig(
        config.model_yaml,
        resume=config.resume,
        use_amp=config.use_amp,
        tuning=config.tuning
    )

    trainer = DetTrainer(cfg)

    ckpt = torch.load(config.resume, map_location='cpu')
    msg = cfg.model.load_state_dict(ckpt)
    print(msg)

    trainer.inference(images_src)
    logger.debug("All images have been successfully processed !")
