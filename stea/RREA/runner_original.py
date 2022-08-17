# -*- coding: utf-8 -*-


import warnings

warnings.filterwarnings('ignore')

import os
import random
import keras
from tqdm import *
import numpy as np
from stea.RREA.utils import *
from stea.RREA.CSLS import *
import tensorflow as tf
import keras.backend as K
from keras.layers import *
from stea.RREA.layer import NR_GraphAttention
import argparse
import json


# parser = argparse.ArgumentParser()
# parser.add_argument('--data_dir', type=str)
# parser.add_argument('--output_dir', type=str)
# parser.add_argument('--device', type=str)
# args = parser.parse_args()
# data_dir = args.data_dir
# output_dir = args.output_dir


# os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"



class TokenEmbedding(keras.layers.Embedding):
    """Embedding layer with weights returned."""

    def compute_output_shape(self, input_shape):
        return self.input_dim, self.output_dim

    def compute_mask(self, inputs, mask=None):
        return None

    def call(self, inputs):
        return self.embeddings



def create_model(batch_size, node_size, rel_size, node_hidden, rel_hidden, triple_size, n_attn_heads=2, dropout_rate=0, gamma=3,
              lr=0.005, depth=2):
    adj_input = Input(shape=(None, 2))
    index_input = Input(shape=(None, 2), dtype='int64')
    val_input = Input(shape=(None,))
    rel_adj = Input(shape=(None, 2))
    ent_adj = Input(shape=(None, 2))

    ent_emb = TokenEmbedding(node_size, node_hidden, trainable=True)(val_input)
    rel_emb = TokenEmbedding(rel_size, node_hidden, trainable=True)(val_input)

    def avg(tensor, size):
        adj = K.cast(K.squeeze(tensor[0], axis=0), dtype="int64")
        adj = tf.SparseTensor(indices=adj, values=tf.ones_like(adj[:, 0], dtype='float32'),
                              dense_shape=(node_size, size))
        adj = tf.sparse_softmax(adj)
        return tf.sparse_tensor_dense_matmul(adj, tensor[1])

    opt = [rel_emb, adj_input, index_input, val_input]
    ent_feature = Lambda(avg, arguments={'size': node_size})([ent_adj, ent_emb])
    rel_feature = Lambda(avg, arguments={'size': rel_size})([rel_adj, rel_emb])

    encoder = NR_GraphAttention(node_size, activation="relu",
                                rel_size=rel_size,
                                depth=depth,
                                attn_heads=n_attn_heads,
                                triple_size=triple_size,
                                attn_heads_reduction='average',
                                dropout_rate=dropout_rate)

    out_feature = Concatenate(-1)([encoder([ent_feature] + opt), encoder([rel_feature] + opt)])
    out_feature = Dropout(dropout_rate)(out_feature)

    # original loss
    alignment_input = Input(shape=(None, 4))
    find = Lambda(lambda x: K.gather(reference=x[0], indices=K.cast(K.squeeze(x[1], axis=0), 'int32')))(
        [out_feature, alignment_input])

    def align_loss(tensor):
        def _cosine(x):
            dot1 = K.batch_dot(x[0], x[1], axes=1)
            dot2 = K.batch_dot(x[0], x[0], axes=1)
            dot3 = K.batch_dot(x[1], x[1], axes=1)
            max_ = K.maximum(K.sqrt(dot2 * dot3), K.epsilon())
            return dot1 / max_

        def l1(ll, rr):
            return K.sum(K.abs(ll - rr), axis=-1, keepdims=True)

        def l2(ll, rr):
            return K.sum(K.square(ll - rr), axis=-1, keepdims=True)

        l, r, fl, fr = [tensor[:, 0, :], tensor[:, 1, :], tensor[:, 2, :], tensor[:, 3, :]]
        loss = K.relu(gamma + l1(l, r) - l1(l, fr)) + K.relu(gamma + l1(l, r) - l1(fl, r))
        return tf.reduce_sum(loss, keep_dims=True) / (batch_size)

    loss = Lambda(align_loss)(find)

    inputs = [adj_input, index_input, val_input, rel_adj, ent_adj]
    train_model = keras.Model(inputs=inputs + [alignment_input], outputs=loss)
    train_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer=keras.optimizers.rmsprop(lr))

    feature_model = keras.Model(inputs=inputs, outputs=out_feature)
    return train_model, feature_model






class Runner():
    def __init__(self, data_dir, output_dir, enhanced=False,
                 max_train_epoch=1200, max_continue_epoch=200, eval_freq=50):
        os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # sess = tf.Session(config=config)

        self.max_train_epoch = max_train_epoch
        self.max_continue_epoch = max_continue_epoch
        self.eval_freq = eval_freq

        self.data_dir = data_dir
        self.output_dir = output_dir

        # lang = "zh"
        # train_pair, dev_pair, adj_matrix, r_index, r_val, adj_features, rel_features = load_data('/home/uqbliu3/experiments/ealib/ext_files/datasets/formatted_rrea/%s_en/'%lang,train_ratio=0.30)
        if enhanced:
            self.train_pair, self.dev_pair, self.adj_matrix, self.r_index, self.r_val, self.adj_features, self.rel_features = load_data3(data_dir)
        else:
            self.train_pair, self.dev_pair, self.adj_matrix, self.r_index, self.r_val, self.adj_features, self.rel_features = load_data2(data_dir)

        self.adj_matrix = np.stack(self.adj_matrix.nonzero(), axis=1)
        self.rel_matrix, self.rel_val = np.stack(self.rel_features.nonzero(), axis=1), self.rel_features.data
        self.ent_matrix, self.ent_val = np.stack(self.adj_features.nonzero(), axis=1), self.adj_features.data

        self.node_size = self.adj_features.shape[0]
        rel_size = self.rel_features.shape[1]
        triple_size = len(self.adj_matrix)
        self.batch_size = self.node_size



        # self.model, self.get_emb = self.get_trgat(dropout_rate=0.30, node_size=self.node_size, rel_size=rel_size, n_attn_heads=1, depth=2,
        #                            gamma=3,
        #                            node_hidden=100, rel_hidden=100, triple_size=triple_size)
        self.model, self.get_emb = create_model(batch_size=self.batch_size, dropout_rate=0.30, node_size=self.node_size, rel_size=rel_size,
                                                  n_attn_heads=1, depth=2,
                                                  gamma=3,
                                                  node_hidden=100, rel_hidden=100, triple_size=triple_size)

        self.model.summary()
        initial_weights = self.model.get_weights()


        self.rest_set_1 = [e1 for e1, e2 in self.dev_pair]
        self.rest_set_2 = [e2 for e1, e2 in self.dev_pair]
        np.random.shuffle(self.rest_set_1)
        np.random.shuffle(self.rest_set_2)

    def get_embedding(self):
        inputs = [self.adj_matrix, self.r_index, self.r_val, self.rel_matrix, self.ent_matrix]
        inputs = [np.expand_dims(item, axis=0) for item in inputs]
        return self.get_emb.predict_on_batch(inputs)

    def test(self, wrank=None):
        vec = self.get_embedding()
        return get_hits(vec, self.dev_pair, wrank=wrank)

    def CSLS_test(self, thread_number=32, csls=10, accurate=True):
        print("processing embeddings")
        vec = self.get_embedding()
        vec = vec / np.linalg.norm(vec, axis=-1, keepdims=True)
        # Lvec = np.array([vec[e1] for e1, e2 in self.dev_pair])
        # Rvec = np.array([vec[e2] for e1, e2 in self.dev_pair])
        dev_alignment = np.array(self.dev_pair)
        Lvec = vec[dev_alignment[:, 0]]
        Rvec = vec[dev_alignment[:, 1]]
        # Lvec = Lvec / np.linalg.norm(Lvec, axis=-1, keepdims=True)
        # Rvec = Rvec / np.linalg.norm(Rvec, axis=-1, keepdims=True)
        print("computing metrics")
        _, hits1, metrics, _ = eval_alignment_by_sim_mat(Lvec, Rvec, [1, 5, 10], thread_number, csls=csls, accurate=accurate)
        return hits1, metrics

    def get_train_set(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        negative_ratio = batch_size // len(self.train_pair) + 1
        train_set = np.reshape(np.repeat(np.expand_dims(self.train_pair, axis=0), axis=0, repeats=negative_ratio),
                               newshape=(-1, 2))
        np.random.shuffle(train_set)
        train_set = train_set[:batch_size]
        train_set = np.concatenate([train_set, np.random.randint(0, self.node_size, train_set.shape)], axis=-1)
        return train_set

    def iterative_training(self):
        logs = {}
        accu_epoch = 0
        if os.path.exists(os.path.join(self.output_dir, "model.ckpt")):
            self.restore_model()
            start_turn = 1
        else:
            start_turn = 0
        for turn in range(start_turn, 5):
            if turn == 0:
                epoch = self.max_train_epoch
            else:
                epoch = 100
            print("iteration %d start." % turn)
            for i in trange(epoch, desc=f"iterative training RREA ({turn})"):
                train_set = self.get_train_set()
                inputs = [self.adj_matrix, self.r_index, self.r_val, self.rel_matrix, self.ent_matrix, train_set]
                inputs = [np.expand_dims(item, axis=0) for item in inputs]
                self.model.train_on_batch(inputs, np.zeros((1, 1)))
                accu_epoch += 1
                if accu_epoch % self.eval_freq == 0:
                    hit1, metrics = self.CSLS_test()
                    logs[accu_epoch] = metrics

            self.save()

            new_pair = []
            vec = self.get_embedding()
            Lvec = np.array([vec[e] for e in self.rest_set_1])
            Rvec = np.array([vec[e] for e in self.rest_set_2])
            Lvec = Lvec / np.linalg.norm(Lvec, axis=-1, keepdims=True)
            Rvec = Rvec / np.linalg.norm(Rvec, axis=-1, keepdims=True)
            A, _, _, _ = eval_alignment_by_sim_mat(Lvec, Rvec, [1, 5, 10], 16, 10, True, False)
            B, _, _, _ = eval_alignment_by_sim_mat(Rvec, Lvec, [1, 5, 10], 16, 10, True, False)
            A = sorted(list(A))
            B = sorted(list(B))
            for a, b in A:
                if B[b][1] == a:
                    new_pair.append([self.rest_set_1[a], self.rest_set_2[b]])
            print("generate new semi-pairs: %d." % len(new_pair))

            self.train_pair = np.concatenate([self.train_pair, np.array(new_pair)], axis=0)
            for e1, e2 in new_pair:
                if e1 in self.rest_set_1:
                    self.rest_set_1.remove(e1)

            for e1, e2 in new_pair:
                if e2 in self.rest_set_2:
                    self.rest_set_2.remove(e2)
        with open(os.path.join(self.output_dir, "training_log.json"), "w+") as file:
            file.write(json.dumps(logs))

    ###  added by Bing ###
    def train(self):
        epoch = 0
        best_perf = None
        logs = {}
        no_improve_num = 0
        # while True:
        for _ in trange(self.max_train_epoch, desc="training RREA"):
            train_set = self.get_train_set()
            inputs = [self.adj_matrix, self.r_index, self.r_val, self.rel_matrix, self.ent_matrix, train_set]
            inputs = [np.expand_dims(item, axis=0) for item in inputs]
            self.model.train_on_batch(inputs, np.zeros((1, 1)))
            epoch += 1
            # if epoch % self.eval_freq == 0:
            #     print(f"# EVALUATION - EPOCH {epoch}:")
            #     hit1, metrics = self.CSLS_test()
            #     logs[epoch] = metrics
            #     if best_perf is None:
            #         best_perf = hit1
            #     elif hit1 > best_perf:
            #         best_perf = hit1
            #         no_improve_num = 0
            #     else:
            #         no_improve_num += 1
            #         if no_improve_num >= 3:
            #             break
        with open(os.path.join(self.output_dir, "training_log.json"), "w+") as file:
            file.write(json.dumps(logs))

    def continue_training(self):
        epoch = 0
        best_perf = None
        log_fn = os.path.join(self.output_dir, "training_log.json")
        if os.path.exists(log_fn):
            with open(log_fn) as file:
                logs = json.loads(file.read())
        else:
            logs = {}
        no_improve_num = 0
        # while True:
        for _ in trange(self.max_continue_epoch, desc="continue training RREA"):
            train_set = self.get_train_set()
            inputs = [self.adj_matrix, self.r_index, self.r_val, self.rel_matrix, self.ent_matrix, train_set]
            inputs = [np.expand_dims(item, axis=0) for item in inputs]
            self.model.train_on_batch(inputs, np.zeros((1, 1)))
            epoch += 1
            # if epoch % self.eval_freq == 0:
            #     print(f"# EVALUATION - EPOCH {epoch}:")
            #     hit1, metrics = self.CSLS_test()
            #     logs[epoch] = metrics
            #     if best_perf is None:
            #         best_perf = hit1
            #     elif hit1 > best_perf:
            #         best_perf = hit1
            #         no_improve_num = 0
            #     else:
            #         no_improve_num += 1
            #         if no_improve_num >= 1:
            #             break
        with open(os.path.join(self.output_dir, "training_log.json"), "w+") as file:
            file.write(json.dumps(logs))


    def save(self, save_metrics=False):
        # save model
        self.model.save_weights(os.path.join(self.output_dir, "model.ckpt"))
        self.get_emb.save_weights(os.path.join(self.output_dir, "get_emb.ckpt"))

        # save embeddings
        vec = self.get_embedding()
        vec = vec / np.linalg.norm(vec, axis=-1, keepdims=True)
        with open(os.path.join(self.data_dir, "ent_ids_1")) as file:
            lines = file.read().strip().split("\n")
            ent1_id_list = [int(line.split()[0]) for line in lines]
        with open(os.path.join(self.data_dir, "ent_ids_2")) as file:
            lines = file.read().strip().split("\n")
            ent2_id_list = [int(line.split()[0]) for line in lines]
        ent1_ids = np.array(ent1_id_list)
        ent2_ids = np.array(ent2_id_list)
        np.savez(os.path.join(self.output_dir, "emb.npz"), embs=vec, ent1_ids=ent1_ids, ent2_ids=ent2_ids)

        # # save metrics
        # if save_metrics:
        #     print("begin evaluating and save metrics")
        #     Lvec = np.array([vec[e1] for e1, e2 in self.dev_pair])
        #     Rvec = np.array([vec[e2] for e1, e2 in self.dev_pair])
        #     csls_test_alignment, _, csls_test_metrics, simi_mat = eval_alignment_by_sim_mat(Lvec, Rvec, [1, 5, 10], nums_threads=100, csls=10,
        #                                                                           accurate=True)
        #     cos_test_alignment, _, cos_test_metrics, simi_mat = eval_alignment_by_sim_mat(Lvec, Rvec, [1, 5, 10], nums_threads=100, csls=0,
        #                                                                         accurate=True)
        #
        #     metrics_obj = {"metrics_csls": csls_test_metrics, "metrics_cos": cos_test_metrics}
        #     # csls_test_alignment = np.array(list(csls_test_alignment), dtype=int)
        #     # cos_test_alignment = np.array(list(cos_test_alignment))
        #     dev_ent2_list = [e2 for e1, e2 in self.dev_pair]
        #     dev_ent1_list = [e1 for e1, e2 in self.dev_pair]
        #     csls_test_alignment = [(int(dev_ent1_list[ent1]), int(dev_ent2_list[rank])) for ent1, rank in list(csls_test_alignment)]
        #     cos_test_alignment = [(int(dev_ent1_list[ent1]), int(dev_ent2_list[rank])) for ent1, rank in list(cos_test_alignment)]
        #     pred_alignment_obj = {"pred_alignment_csls": csls_test_alignment,
        #                           "pred_alignment_cos": cos_test_alignment}
        #     with open(os.path.join(self.output_dir, "metrics.json"), "w+") as file:
        #         file.write(json.dumps(metrics_obj))
        #     with open(os.path.join(self.output_dir, "pred_alignment.json"), "w+") as file:
        #         file.write(json.dumps(pred_alignment_obj))


    def restore_model(self, from_dir=None):
        if from_dir is None:
            from_dir = self.output_dir
        if os.path.exists(os.path.join(from_dir, "model.ckpt")):
            self.model.load_weights(os.path.join(from_dir, "model.ckpt"))
            self.get_emb.load_weights(os.path.join(from_dir, "get_emb.ckpt"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--max_train_epoch', type=int)
    parser.add_argument('--max_continue_epoch', type=int)
    parser.add_argument('--eval_freq', type=int, default=10000)
    parser.add_argument('--initial_training', type=str)
    # parser.add_argument('--neu_save_metrics', type=int)
    parser.add_argument('--enhanced', default=False, type=bool)
    parser.add_argument('--restore_from_dir', default="", type=str)
    args = parser.parse_args()

    runner = Runner(data_dir=args.data_dir,
                    output_dir=args.output_dir,
                    max_train_epoch=args.max_train_epoch,
                    max_continue_epoch=args.max_continue_epoch,
                    eval_freq=args.eval_freq,
                    enhanced=args.enhanced
                    )

    if args.enhanced:
        runner.restore_model(args.restore_from_dir)
        runner.continue_training()
    else:
        if args.initial_training in ["supervised", "sup" ]:
            runner.train()
        elif args.initial_training in  ["iterative", "semi"]:
            runner.iterative_training()
        else:
            raise Exception("unknown initial training method")

    runner.save()


