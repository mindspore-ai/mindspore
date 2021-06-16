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
"""post process for 310 inference"""
import os
import sys
import six
import lmdb
from PIL import Image
from src.model_utils.config import config
from src.util import CTCLabelConverter


def get_img_from_lmdb(env_, ind):
    with env_.begin(write=False) as txn_:
        label_key = 'label-%09d'.encode() % ind
        label_ = txn_.get(label_key).decode('utf-8')
        img_key = 'image-%09d'.encode() % ind
        imgbuf = txn_.get(img_key)

        buf = six.BytesIO()
        buf.write(imgbuf)
        buf.seek(0)
        try:
            img_ = Image.open(buf).convert('RGB')  # for color image

        except IOError:
            print(f'Corrupted image for {ind}')
            # make dummy image and dummy label for corrupted image.
            img_ = Image.new('RGB', (config.IMG_W, config.IMG_H))
            label_ = '[dummy_label]'

    label_ = label_.lower()

    return img_, label_


if __name__ == '__main__':
    max_len = int((26 + 1) // 2)
    converter = CTCLabelConverter(config.CHARACTER)
    env = lmdb.open(config.TEST_DATASET_PATH, max_readers=32, readonly=True, lock=False, readahead=False, meminit=False)
    if not env:
        print('cannot create lmdb from %s' % (config.TEST_DATASET_PATH))
        sys.exit(0)

    with env.begin(write=False) as txn:
        nSamples = int(txn.get('num-samples'.encode()))
        nSamples = nSamples

        # Filtering
        filtered_index_list = []
        for index_ in range(nSamples):
            index_ += 1  # lmdb starts with 1
            label_key_ = 'label-%09d'.encode() % index_
            label = txn.get(label_key_).decode('utf-8')

            if len(label) > max_len:
                continue

            illegal_sample = False
            for char_item in label.lower():
                if char_item not in config.CHARACTER:
                    illegal_sample = True
                    break
            if illegal_sample:
                continue

            filtered_index_list.append(index_)

    img_ret = []
    text_ret = []

    print(f'num of samples in IIIT dataset: {len(filtered_index_list)}')
    i = 0
    label_dict = {}
    for index in filtered_index_list:
        img, label = get_img_from_lmdb(env, index)
        img_name = os.path.join(config.preprocess_output, str(i) + ".png")
        img.save(img_name)
        label_dict[str(i)] = label
        i += 1
    with open('./label.txt', 'w') as file:
        for k, v in label_dict.items():
            file.write(str(k) + ',' + str(v) + '\n')
