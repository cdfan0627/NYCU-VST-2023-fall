#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.
import os

from yolox.exp import Exp as MyExp


class Exp(MyExp):
    def __init__(self):
        super(Exp, self).__init__()
        self.depth = 0.33
        self.width = 0.50
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        #self.input_size = (224, 224)
        # Define yourself dataset path
        self.data_dir = "datasets/car_coco"
        self.train_ann = "train_labels.json"
        self.val_ann = "val_labels.json"
        #self.seed = 1325
        self.num_classes = 1
       
