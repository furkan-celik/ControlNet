from share import *

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from tutorial_dataset import MyDataset
from cldm.logger import ImageLogger
from cldm.model import create_model, load_state_dict
from wui import build_wui_dsets


# Configs
resume_path = './models/control_sd15_ini.ckpt'
batch_size = 4
logger_freq = 300
learning_rate = 1e-5
sd_locked = True
only_mid_control = False


# First use cpu to load models. Pytorch Lightning will automatically move it to GPUs.
model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
c = load_state_dict(resume_path, location='cpu')
del c['control_model.input_hint_block.0.weight']
del c['control_model.input_hint_block.0.bias']
model.load_state_dict(c, strict=False)
model.learning_rate = learning_rate
model.sd_locked = sd_locked
model.only_mid_control = only_mid_control


# Misc
dataloader = build_wui_dsets({
            "split_file": "/content/balanced_7k.json",
            "boxes_dir": "/content/webui-boxes/all_data",
            "rawdata_screenshots_dir": "/content/ds_all",
            "class_map_file": "/content/controlnet/class_map.json",
            "max_boxes": 100,
            "layout_length": 100,
            "num_classes_for_layout_object": 82,
            "mask_size_for_layout_object": 128,
            "used_condition_types": [
                "obj_class",
                "obj_bbox"
            ],
            "image_size": 256
        }, 16)
# dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
logger = ImageLogger(batch_frequency=logger_freq)
trainer = pl.Trainer(gpus=1, precision=16, callbacks=[logger])


# Train!
trainer.fit(model, dataloader)
