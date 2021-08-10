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
"""start create_dataset_ILSVRC"""
import pickle
import os
import functools
import xml.etree.ElementTree as ET
import sys
import argparse
import multiprocessing
from multiprocessing import Pool
from glob import glob
import cv2
from tqdm import tqdm
from src import config, get_instance_image
sys.path.append(os.getcwd())
multiprocessing.set_start_method('spawn', True)

def worker(output_dir, video_dir):
    """worker used read image and box position"""
    image_names = glob(os.path.join(video_dir, '*.JPEG'))
    video_name = video_dir.split('/')[-1]
    save_folder = os.path.join(output_dir, video_name)
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    trajs = {}
    for image_name in image_names:
        img = cv2.imread(image_name)
        img_mean = tuple(map(int, img.mean(axis=(0, 1))))
        anno_name = image_name.replace('Data', 'Annotations')
        anno_name = anno_name.replace('JPEG', 'xml')
        tree = ET.parse(anno_name)
        root = tree.getroot()

        filename = root.find('filename').text
        for obj in root.iter('object'):
            bbox = obj.find(
                'bndbox')
            bbox = list(map(int, [bbox.find('xmin').text,
                                  bbox.find('ymin').text,
                                  bbox.find('xmax').text,
                                  bbox.find('ymax').text]))
            trkid = int(obj.find('trackid').text)
            if trkid in trajs:
                trajs[trkid].append(filename)
            else:
                trajs[trkid] = [filename]
            instance_img, _, _ = get_instance_image(img, bbox,
                                                    config.exemplar_size,
                                                    config.instance_size,
                                                    config.context_amount,
                                                    img_mean)
            instance_img_name = os.path.join(save_folder, filename + ".{:02d}.x.jpg".format(trkid))
            cv2.imwrite(instance_img_name, instance_img)
    return video_name, trajs

def processing(data_dir, output_dir, num_threads=32):
    """
    the mian process to pretreatment picture and use multi-threads
    """
    video_dir = os.path.join(data_dir, 'Data/VID')
    all_videos = glob(os.path.join(video_dir, 'train/ILSVRC2015_VID_train_0000/*')) + \
                 glob(os.path.join(video_dir, 'train/ILSVRC2015_VID_train_0001/*')) + \
                 glob(os.path.join(video_dir, 'train/ILSVRC2015_VID_train_0002/*')) + \
                 glob(os.path.join(video_dir, 'train/ILSVRC2015_VID_train_0003/*')) + \
                 glob(os.path.join(video_dir, 'val/*'))
    meta_data = []
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with Pool(processes=num_threads) as pool:
        for ret in tqdm(pool.imap_unordered(functools.partial(worker, output_dir), all_videos),
                        total=len(all_videos)):
            meta_data.append(ret)
    pickle.dump(meta_data, open(os.path.join(output_dir, "meta_data.pkl"), 'wb'))
Data_dir = '/data/VID/ILSVRC2015'
Output_dir = '/data/VID/ILSVRC_VID_CURATION_train'
Num_threads = 32

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Demo SiamFC")
    parser.add_argument('--d', default=Data_dir, type=str, help="data_dir")
    parser.add_argument('--o', default=Output_dir, type=str, help="out put")
    parser.add_argument('--t', default=Num_threads, type=int, help="thread_num")
    args = parser.parse_args()
    processing(args.d, args.o, args.t)
