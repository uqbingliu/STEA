# -*- coding: utf-8 -*-

import torch
from stea.conf import Config
import os
from stea.data import load_alignment
import copy
from stea.components_base import JointDistrModule, NeuralEAModule, evaluate_models

import json
import numpy as np
from stea.RREA.CSLS_torch import Evaluator
import gc



class STEAFramework:
    def __init__(self, conf: Config, jointdistr_model: JointDistrModule, jointdistr_model_inv: JointDistrModule, neural_ea_model: NeuralEAModule):
        self.conf = conf
        self.joint_distri_model = jointdistr_model
        self.joint_distri_model_inv = jointdistr_model_inv
        self.neural_ea_model = neural_ea_model

        self.cache_neural_prob_mtx = None
        self.cache_improved_prob_mtx = None

        self.test_alignment = load_alignment(os.path.join(conf.data_dir, "test_alignment.txt"))

        self.evaluator = Evaluator(self.conf.device)

    def initialize(self):
        print(self.conf.neural_mn)
        self._update_neu_output_dir(0)
        self.neural_ea_model.train_model_with_observed_labels()
        print("## EVALUATING initial neural model ...")

        self.neural_ea_model.evaluate()
        print(self.conf.neural_mn)



    def _update_jointdistr_output_dir(self, ite_no):
        joint_conf = copy.deepcopy(self.conf)
        joint_conf.output_dir = os.path.join(self.conf.output_dir, f"iteration{ite_no}/joint/")
        self.joint_distri_model.conf = joint_conf
        # tmp_conf.paris_cache_dir = os.path.join(self.conf.output_dir, "paris_cache")
        if not os.path.exists(joint_conf.output_dir):
            os.makedirs(joint_conf.output_dir)
        if self.joint_distri_model_inv is not None:
            joint_conf = copy.deepcopy(self.conf)
            joint_conf.output_dir = os.path.join(self.conf.output_dir, f"iteration{ite_no}/joint_inv/")
            self.joint_distri_model_inv.conf = joint_conf
            # tmp_conf.paris_cache_dir = os.path.join(self.conf.output_dir, "paris_cache")
            if not os.path.exists(joint_conf.output_dir):
                os.makedirs(joint_conf.output_dir)

    def _update_neu_output_dir(self, ite_no):
        neu_conf = copy.deepcopy(self.conf)
        neu_conf.output_dir = os.path.join(self.conf.output_dir, f"iteration{ite_no}/neu/")
        if ite_no > 0:
            neu_conf.restore_from_dir = os.path.join(self.conf.output_dir, f"iteration{ite_no - 1}/neu/")
        else:
            neu_conf.restore_from_dir = os.path.join(self.conf.output_dir, f"iteration{ite_no}/neu/")
        self.neural_ea_model.conf = neu_conf
        if not os.path.exists(neu_conf.output_dir):
            os.makedirs(neu_conf.output_dir)


    def run_EM(self):
        import time
        time_log = {}
        print("====>>>> INITIALIZING neural model ...")
        t1 = time.time()
        self.initialize()
        t2 = time.time()
        time_log["initialize"] = t2-t1
        print(f"time for initialization: {(t2-t1)/60} min")
        # self._update_neu_output_dir(0)
        for ite in range(1, self.conf.em_iteration_num+1):
            print(self.conf.neural_mn)
            print(f"====>>>> BEGIN iteration {ite}")
            print("===>> Perform M Step <<===")
            # M step
            print("set staff of neural model")
            t3 = time.time()
            
            neural_sim_mtx=None
            neural_prob_mtx=None
            neural_prob_mtx_inv=None
            if self.conf.no_joint_distr == "True":
                neural_sim_mtx= self.neural_ea_model.predict_simi()
            if self.joint_distri_model is not None:
                neural_prob_mtx, neural_prob_mtx_inv = self.neural_ea_model.predict()
                self.joint_distri_model.set_stuff_about_neu_model(prob_mtx=neural_prob_mtx)
                if self.joint_distri_model_inv is not None:
                    self.joint_distri_model_inv.set_stuff_about_neu_model(prob_mtx=neural_prob_mtx_inv)


            if self.joint_distri_model is not None:
                self._update_jointdistr_output_dir(ite)
                # print("train joint distri")
                # self.joint_distri_model.train_model()
                # if self.joint_distri_model_inv is not None:
                #     print("train inverse joint distri")
                #     self.joint_distri_model_inv.train_model()

            # # E step
            print("===>> Perform E Step <<===")
            if self.joint_distri_model is not None:
                improved_prob_mtx = self.joint_distri_model.coordinate_ascend()
                del self.joint_distri_model.neural_prob_mtx
                if self.joint_distri_model_inv is not None:
                    improved_prob_mtx_inv = self.joint_distri_model_inv.coordinate_ascend()
                    del self.joint_distri_model_inv.neural_prob_mtx
            else:
                improved_prob_mtx = None
                improved_prob_mtx_inv = None
            
            
            t4 = time.time()
            time_log[f"joint_iteration_{ite}"] = t4 - t3

            t5 = time.time()
            
            if self.joint_distri_model is not None:
                print("## EVALUATING improved prob (coordinate ascent)")
                # joint_metrics = evaluate_models(improved_prob_mtx.cpu().numpy(), self.test_alignment)
                joint_metrics, _ = self.evaluator.compute_metrics(improved_prob_mtx, self.test_alignment)
                
                print(joint_metrics)
                
                if self.joint_distri_model_inv is not None:
                    print("## EVALUATING inv improved prob (coordinate ascent)")
                    joint_metrics_inv, _ = self.evaluator.compute_metrics(improved_prob_mtx_inv, self.test_alignment)
                    print(joint_metrics_inv)           
            
            

            
                with open(os.path.join(self.joint_distri_model.conf.output_dir, "eval_metrics.json"), "w+") as file:
                    file.write(json.dumps(joint_metrics))
                if self.joint_distri_model_inv is not None:
                    with open(os.path.join(self.joint_distri_model_inv.conf.output_dir, "eval_metrics.json"), "w+") as file:
                        file.write(json.dumps(joint_metrics_inv))
            else:
                improved_prob_mtx=None
                improved_prob_mtx_inv=None

            t6 = time.time()
            self._update_neu_output_dir(ite)
            
            print("## enhancing labels with joint distribution model")
            self.neural_ea_model.enhance_latent_labels_with_jointdistr(
                improved_prob_mtx,improved_prob_mtx_inv, neural_sim_mtx,None,
                conf=self.conf)
            
            del neural_prob_mtx
            del neural_prob_mtx_inv
            # release memory for prob_mtx (improved_prob_mtx and neural_prob_mtx actually share the same memory)
            del improved_prob_mtx
            del improved_prob_mtx_inv
            print("## training embedding model")
            self.neural_ea_model.train_model_with_observed_n_latent_labels(ite)
            gc.collect()
            
            
            t7 = time.time()
            time_log[f"neural_iteration_{ite}"] = t7 - t6

            print("## EVALUATING updated neural model")
            self.neural_ea_model.evaluate()

            t8 = time.time()

            print(f"time for training joint distr: {(t4 - t3) / 60} min")
            print(f"time for coordinate ascent: {(t5 - t4) / 60} min")
            print(f"time for evaluate improved prob: {(t6 - t5) / 60} min")
            print(f"time for training neural model: {(t7 - t6) / 60} min")
            print(f"time for evaluating neural model: {(t8 - t7) / 60} min")
            if not os.path.exists(self.conf.res_dir):
                os.mkdir(self.conf.res_dir)
            with open(os.path.join(self.conf.res_dir, "time_log.json"), "w+") as file:
                file.write(json.dumps(time_log))

    def evaluate(self):
        print("====>>> Evaluate EM ")
        metrics = {}
        for ite in range(0, self.conf.em_iteration_num + 1):
            print(f"## Evaluate updated neural model (iteration {ite})")
            self._update_neu_output_dir(ite)
            metric_fn = os.path.join(self.neural_ea_model.conf.output_dir, "eval_metrics.json")
            if os.path.exists(metric_fn):
                with open(metric_fn) as file:
                    neu_metrics = json.loads(file.read())
            else:
                with torch.no_grad():
                    neural_prob_mtx = self.neural_ea_model.predict()
                neu_metrics = evaluate_models(neural_prob_mtx.cpu().numpy(), self.test_alignment)
                with open(metric_fn, "w+") as file:
                    file.write(json.dumps(neu_metrics))
            metrics[f"iteration{ite}"] = {"neu": neu_metrics}
        if self.joint_distri_model is not None:
            for ite in range(1, self.conf.em_iteration_num + 1):
                print(f"## Evaluate improved prob (coordinate ascent) (iteration {ite})")
                self._update_jointdistr_output_dir(ite)
                metric_fn = os.path.join(self.joint_distri_model.conf.output_dir, "eval_metrics.json")
                if os.path.exists(metric_fn):
                    with open(metric_fn) as file:
                        joint_metrics = json.loads(file.read())
                else:
                    # improved_prob_mtx = self.joint_distri_model.coordinate_ascend()
                    improved_prob_mtx = np.load(os.path.join(self.joint_distri_model.conf.output_dir, "coordinate_ascent_prob_mtx.npz"))["improved_prob_mtx"]
                    joint_metrics = evaluate_models(improved_prob_mtx, self.test_alignment)
                    with open(metric_fn, "w+") as file:
                        file.write(json.dumps(joint_metrics))
                metrics[f"iteration{ite}"]["joint"] = joint_metrics
                
            for ite in range(1, self.conf.em_iteration_num + 1):
                print(f"## Evaluate inv improved prob (coordinate ascent) (iteration {ite})")
                self._update_jointdistr_output_dir(ite)
                metric_fn = os.path.join(self.joint_distri_model_inv.conf.output_dir, "eval_metrics.json")
                if os.path.exists(metric_fn):
                    with open(metric_fn) as file:
                        joint_metrics = json.loads(file.read())
                else:
                    # improved_prob_mtx = self.joint_distri_model.coordinate_ascend()
                    improved_prob_mtx = np.load(os.path.join(self.joint_distri_model_inv.conf.output_dir, "coordinate_ascent_prob_mtx.npz"))["improved_prob_mtx"]
                    joint_metrics = evaluate_models(improved_prob_mtx, self.test_alignment)
                    with open(metric_fn, "w+") as file:
                        file.write(json.dumps(joint_metrics))
                metrics[f"iteration{ite}"]["joint_inv"] = joint_metrics
        with open(os.path.join(self.conf.res_dir, "metrics.json"), "w+") as file:
            file.write(json.dumps(metrics))





