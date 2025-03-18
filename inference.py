from share import *
import config

import cv2
import einops
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from wui import build_wui_dsets


apply_canny = CannyDetector()

model = create_model('./models/cldm_v15.yaml').cpu()
model.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location='cuda'))
model = model.cuda()
ddim_sampler = DDIMSampler(model)

dataloader = build_wui_dsets({
    "split_file": "/content/balanced_7k.json",
    "boxes_dir": "/content/webui-boxes/all_data",
    "rawdata_screenshots_dir": "/content/ds_all",
    "class_map_file": "/content/layout2im/class_map.json",
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

with torch.no_grad():
    for batch in dataloader:
        text = batch["txt"]
        hint = batch["hint"]

        cond = {
            "c_concat": [hint], 
            "c_crossattn": [model.get_learned_conditioning([text])]
        }
        un_cond = {
            "c_concat": [hint],
            "c_crossattn": [model.get_learned_conditioning([[""] * 100] * len(text))]
        }

        shape = (4, batch["jpg"].shape[2] // 8, batch["jpg"].shape[3] // 8)
        # model.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler.sample(1000, len(hint),
                                                     shape, cond, verbose=False,
                                                     unconditional_guidance_scale=2.5,
                                                     unconditional_conditioning=un_cond)