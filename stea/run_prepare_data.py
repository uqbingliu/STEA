# -*- coding: utf-8 -*-


from stea.data import prepare_data_for_rrea
from stea.data import prepare_data_for_joint_distr
from stea.data import prepare_data_for_openea
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str)
    parser.add_argument('--data_name', type=str)
    parser.add_argument('--train_percent',default=0.3, type=float)
    parser.add_argument('--for_openea', action='store_true', default=False)
    args = parser.parse_args()

    data_dir = args.data_dir
    train_percent = args.train_percent
    data_name = args.data_name

    prepare_data_for_rrea(data_dir, train_percent)
    prepare_data_for_joint_distr(data_dir, data_name)

    if args.for_openea:
        prepare_data_for_openea(data_dir)

    print("complete preparing data")

