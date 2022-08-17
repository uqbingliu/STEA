# -*- coding: utf-8 -*-


import yaml


class Config:
    def __init__(self, fn=None):
        # files and directory
        self.data_dir = None
        self.data_name = None
        self.output_dir = None
        self.res_dir = None

        self.percent_of_training_data = 0.3
        # running env
        self.tf_device = None
        self.device = "cuda:0"
        self.second_device = "cpu"
        self.py_exe_fn = None

        # EMEA
        self.topK = None
        self.cross_entropy_tau = None
        self.restore_from_dir = None
        self.conflict_checker_neigh_num = None
        self.em_iteration_num = None
        self.initial_training = None
        self.continue_training = None
        self.max_train_epoch = None
        self.max_continue_epoch = None
        self.ruleset = None

        # self.eval_freq = None
        self.neu_save_metrics = None

        # openea
        self.openea_script_fn = None
        self.openea_arg_fn = None
        # self.ruleset = None

        # at the bottom of init
        if fn is not None:
            with open(fn) as file:
                conf = yaml.load(file, Loader=yaml.FullLoader)
                self.__dict__.update(conf)


    def update(self, conf_dict: dict):
        self.__dict__.update(conf_dict)




