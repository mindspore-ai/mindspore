# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""pre process for 310 inference"""
import os
from os.path import join

import argparse
import numpy as np

selected_attrs = [
    'Bald', 'Bangs', 'Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Bushy_Eyebrows',
    'Eyeglasses', 'Male', 'Mouth_Slightly_Open', 'Mustache', 'No_Beard', 'Pale_Skin', 'Young'
]

def parse(arg=None):
    """Define configuration of preprocess"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--attrs', dest='attrs', default=selected_attrs, nargs='+', help='attributes to learn')
    parser.add_argument('--attrs_path', type=str, default='../data/list_attr_custom.txt')
    parser.add_argument('--test_int', dest='test_int', type=float, default=1.0)
    parser.add_argument('--thres_int', dest='thres_int', type=float, default=0.5)
    return parser.parse_args(arg)

args = parse()
args.n_attrs = len(args.attrs)

def check_attribute_conflict(att_batch, att_name, att_names):
    """Check Attributes"""
    def _set(att, att_name):
        if att_name in att_names:
            att[att_names.index(att_name)] = 0.0

    att_id = att_names.index(att_name)
    for att in att_batch:
        if att_name in ['Bald', 'Receding_Hairline'] and att[att_id] != 0:
            _set(att, 'Bangs')
        elif att_name == 'Bangs' and att[att_id] != 0:
            _set(att, 'Bald')
            _set(att, 'Receding_Hairline')
        elif att_name in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair'] and att[att_id] != 0:
            for n in ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Gray_Hair']:
                if n != att_name:
                    _set(att, n)
        elif att_name in ['Straight_Hair', 'Wavy_Hair'] and att[att_id] != 0:
            for n in ['Straight_Hair', 'Wavy_Hair']:
                if n != att_name:
                    _set(att, n)
        elif att_name in ['Mustache', 'No_Beard'] and att[att_id] != 0:
            for n in ['Mustache', 'No_Beard']:
                if n != att_name:
                    _set(att, n)
    return att_batch

def read_cfg_file(attr_path):
    """Read configuration from attribute file"""
    attr_list = open(attr_path, "r", encoding="utf-8").readlines()[1].split()
    atts = [attr_list.index(att) + 1 for att in selected_attrs]
    labels = np.loadtxt(attr_path, skiprows=2, usecols=atts, dtype=np.int)
    attr_number = int(open(attr_path, "r", encoding="utf-8").readlines()[0])
    labels = [labels] if attr_number == 1 else labels[0:]
    new_attr = []
    for index in range(attr_number):
        att = [np.asarray((labels[index] + 1) // 2)]
        new_attr.append(att)
    new_attr = np.array(new_attr)
    return new_attr, attr_number

def preprocess_cfg(attrs, numbers):
    """Preprocess attribute file"""
    new_attr = []
    for index in range(numbers):
        attr = attrs[index]
        att_b_list = [attr]
        for i in range(args.n_attrs):
            tmp = attr.copy()
            tmp[:, i] = 1 - tmp[:, i]
            tmp = check_attribute_conflict(tmp, selected_attrs[i], selected_attrs)
            att_b_list.append(tmp)
        for i, att_b in enumerate(att_b_list):
            att_b_ = (att_b * 2 - 1) * args.thres_int
            if i > 0:
                att_b_[..., i - 1] = att_b_[..., i - 1] * args.test_int / args.thres_int
            new_attr.append(att_b_)
    return new_attr

def write_cfg_file(attrs, numbers):
    """Write attribute file"""
    cur_dir = os.getcwd()
    print(cur_dir)
    path = join(cur_dir, 'attrs.txt')
    with open(path, "w") as f:
        f.writelines(str(numbers))
        f.writelines("\n")
        f.writelines(str(args.n_attrs))
        f.writelines("\n")
        counts = numbers * args.n_attrs
        for index in range(counts):
            attrs_list = attrs[index][0]
            new_attrs_list = ["%s" % x for x in attrs_list]
            sequence = " ".join(new_attrs_list)
            f.writelines(sequence)
            f.writelines("\n")
    print("Generate cfg file successfully.")

if __name__ == "__main__":

    if args.attrs_path is None:
        print("Path is not correct!")
    attributes, n_images = read_cfg_file(args.attrs_path)
    new_attrs = preprocess_cfg(attributes, n_images)
    write_cfg_file(new_attrs, n_images)
