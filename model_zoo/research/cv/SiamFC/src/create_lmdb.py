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
"""start create lmdb"""
import os
import hashlib
import functools
import argparse
from glob import glob
import multiprocessing
from multiprocessing import Pool
from tqdm import tqdm
import lmdb
import cv2
multiprocessing.set_start_method('spawn', True)

def worker(video_name):
    """
    workers used create key and value
    """
    image_names = glob(video_name + '/*')
    kv = {}
    for image_name in image_names:
        img = cv2.imread(image_name)
        _, img_encode = cv2.imencode('.jpg', img)
        img_encode = img_encode.tobytes()
        kv[hashlib.md5(image_name.encode()).digest()] = img_encode
    return kv


def create_lmdb(data_dir, output_dir, num_threads):
    """
    create lmdb use multi-threads
    """
    video_names = glob(data_dir + '/*')
    video_names = [x for x in video_names if os.path.isdir(x)]
    db = lmdb.open(output_dir, map_size=int(50e12))
    with Pool(processes=num_threads) as pool:
        for ret in tqdm(pool.imap_unordered(functools.partial(worker),
                                            video_names),
                        total=len(video_names)):
            with db.begin(write=True) as txn:
                for k, v in ret.items():
                    txn.put(k, v)

Data_dir = '/data/VID/ILSVRC_VID_CURATION_train'
Output_dir = '/data/VID/ILSVRC_VID_CURATION_train.lmdb'
Num_threads = 32
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Demo SiamFC")
    parser.add_argument('--d', default=Data_dir, type=str, help="data_dir")
    parser.add_argument('--o', default=Output_dir, type=str, help="out put")
    parser.add_argument('--n', default=Num_threads, type=int, help="thread_num")
    args = parser.parse_args()

    create_lmdb(args.d, args.o, args.n)
