import argparse
import concurrent
import itertools
import json
import os
import pathlib
import random
import shutil
from datetime import datetime

import numpy as np
import pandas as pd
import torch
from joblib import Parallel, delayed

from exp import Exp


def get_device(device):
    if device == 'cuda':
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return torch.device('cpu')


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def print_args(args, n=2, k=0, v=0):
    if k == 0:
        k = max([len(k) for k, v in vars(args).items()]) + 4
    if v == 0:
        v = max([len(str(v)) for k, v in vars(args).items()]) + 4
    items = list(vars(args).items())
    items.sort(key=lambda x: x[0])
    for i in range(0, len(items), n):
        line = ""
        for j in range(n):
            if i + j < len(items):
                # 获取键值对
                key, value = items[i + j]
                # 格式化键值对，并设置颜色
                line += f"| \033[92m {key:<{k}} \033[94m{str(value):>{v}} \033[0m"
        line += "|"
        print(line)


def parse_args():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--enhance', type=str, default=None)
    parser.add_argument('--uci_id', type=int, default=53)
    parser.add_argument('--train_ratio', type=float, default=0.4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch', type=int, default=32)
    parser.add_argument('--hidden', type=int, default=32)
    parser.add_argument('--layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--lr', type=float, default=0.001)
    # parser.add_argument('--num_workers', type=int, default=2)
    # parser.add_argument('--prefetch_factor', type=int, default=5)
    # parser.add_argument('--persistent_workers', type=bool, default=False)
    parser.add_argument('--comment', type=str, default='')

    parser.add_argument('--white_noise_level', type=float, default=0.1)
    parser.add_argument('--r', type=float, default=0.1)
    parser.add_argument('--salt_noise_ratio', type=float, default=0.1)
    parser.add_argument('--mask_ratio', type=float, default=0.1)

    args = parser.parse_args()
    return args


def save_args_to_json(args, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=4)


def copy_args(args):
    return argparse.Namespace(**vars(args))


def run(args, r_path=None):
    if r_path:
        hist_args_list = get_args_list(r_path)
        for hist_args in hist_args_list:
            if dict_equals(vars(args), hist_args):
                print('已经运行过了')
                return

    set_seed(args.seed)
    exp = Exp(args)
    exp.exp_init()
    exp.train()
    exp.test()
    exp.save()


def cartesian_product_dict(test_space, key_order=None):
    if key_order is None:
        key_order = list(test_space.keys())

    values_combinations = itertools.product(*(test_space[key] for key in key_order))

    result = []
    for combination in values_combinations:
        result.append(dict(zip(key_order, combination)))

    return result


def dict_equals(d_a, d_b):
    keys = set(d_a).intersection(d_b)
    for k in keys:
        if d_a[k] != d_b[k]:
            # print(f"{k}: {d_a[k]} {d_b[k]}")
            return False
    return True


def filter_unrun_args_list(cur_args_list, hist_args_list):
    res = []
    for cur_args in cur_args_list:
        args = vars(cur_args)
        for hist_args in hist_args_list:
            if dict_equals(args, hist_args):
                break
        else:
            res.append(cur_args)
    return res


def filter_setting_worker(folder, r_path):
    if not check_folder(folder):
        delete_folder(r_path, folder)


def check_folder(folder, file_name='args.json', silent=True):
    args_path = folder / file_name
    if not silent:
        print(args_path, ', exist: ', args_path.exists())
    return args_path.exists()


def delete_folder(path, folder):
    # 提取folder的name,删除path下的name文件夹
    path = pathlib.Path(path)
    folder_name = folder.name
    folder_to_delete = path / folder_name
    if folder_to_delete.exists() and folder_to_delete.is_dir():
        shutil.rmtree(folder_to_delete)
        print(f'delete {folder_to_delete}')


def filter_setting(r_path):
    if not os.path.exists(r_path):
        return
    folders = get_settings(r_path)
    worker = 4
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker) as executor:
        executor.map(filter_setting_worker, folders, [r_path] * len(folders))


def get_args_list(r_path):
    if not os.path.exists(r_path):
        return []
    folders = get_settings(r_path)
    worker = 4
    with concurrent.futures.ProcessPoolExecutor(max_workers=worker) as executor:
        args_list = list(executor.map(get_args_worker, folders))
    return [args for args in args_list if args is not None]


def get_args_worker(folder):
    args_json_path = folder / 'args.json'
    if args_json_path.exists():
        with open(args_json_path, 'r') as f:
            args = json.load(f)
        return args
    else:
        print(args_json_path, 'not exist')
        return None


def reload_args(cur_args_list, r_path=None):
    if not r_path:
        r_path = './out/'
    filter_setting(r_path)
    args_list = get_args_list(r_path)
    res = filter_unrun_args_list(cur_args_list, args_list)
    print(f'load {len(cur_args_list) - len(res)} args from {r_path}, remain {len(res)} args')
    return res


def get_arg_list(args, combine_space=None, alter_space=None, r_path=None, sf=False):
    arg_list = [args]
    if combine_space:
        new_arg_list = []
        for cur_args in arg_list:
            for args_dict in cartesian_product_dict(combine_space):
                new_args = argparse.Namespace(**vars(cur_args))
                for k, v in args_dict.items():
                    setattr(new_args, k, v)
                new_arg_list.append(new_args)
        arg_list = new_arg_list

    if alter_space:
        new_arg_list = []
        for cur_args in arg_list:
            for key, values in alter_space.items():
                for value in values:
                    new_args = argparse.Namespace(**vars(cur_args))
                    setattr(new_args, key, value)
                    new_arg_list.append(new_args)
        arg_list = new_arg_list

    arg_list = reload_args(arg_list, r_path)
    if sf:
        random.shuffle(arg_list)

    print(f'Total {len(arg_list)} args')

    return arg_list


def get_settings(r_path):
    r_path = pathlib.Path(r_path)
    folders = [folder for folder in r_path.iterdir() if folder.is_dir()]
    return folders


def get_file_creation_time(file_path):
    path = pathlib.Path(file_path)
    timestamp = path.stat().st_ctime
    creation_time = datetime.fromtimestamp(timestamp)
    return creation_time.strftime("%Y/%m/%d-%H/%M/%S")


def settings2csv(r_path=None, out_path=None, keys=None):
    if not r_path:
        r_path = './out'
    settings = get_settings(r_path)
    filter_settings = [setting for setting in settings if (setting / 'args.json').exists()]
    res = []
    for setting in filter_settings:
        args_path = setting / 'args.json'
        with open(args_path, 'r') as f:
            args = json.load(f)
        if keys is not None:
            args = {k: v for k, v in args.items() if k in keys}
        row = list(args.values())
        row.append(setting.resolve())
        row.append(get_file_creation_time(args_path))
        res.append(row)
    col = list(args.keys()) + ['settings', 'complete_time']
    df = pd.DataFrame(res, columns=col)

    # 按照 col 中的所有列进行排序
    df_sorted = df.sort_values(by=keys if keys else col)
    df_sorted.index = range(1, len(df_sorted) + 1)

    if out_path is None:
        out_path = os.path.join(r_path, 'result.csv')
    else:
        out_path = os.path.join(out_path, 'result.csv')
    df_sorted.to_csv(out_path)
    return df


def parallel_run(args_list, workers=4):
    Parallel(n_jobs=workers, prefer='processes', backend='loky')(delayed(run)(args) for args in args_list)
