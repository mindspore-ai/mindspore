# Copyright 2020 Huawei Technologies Co., Ltd
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
"""Generate train and test dataset"""
import os
import shutil
import math as m
import random
from multiprocessing import Process
from captcha.image import ImageCaptcha


def _generate_captcha_per_process(path, total, start, end, img_width, img_height, max_digits):
    captcha = ImageCaptcha(width=img_width, height=img_height)
    filename_head = '{:0>' + str(len(str(total))) + '}-'
    for i in range(start, end):
        digits = ''
        digits_length = random.randint(1, max_digits)
        for _ in range(0, digits_length):
            integer = random.randint(0, 9)
            digits += str(integer)
        captcha.write(digits, os.path.join(path, filename_head.format(i) + digits + '.png'))


def generate_captcha(name, img_num, img_width, img_height, max_digits, process_num=16):
    """
    generate captcha images

    Args:
        name(str): name of folder, under which captcha images are saved in
        img_num(int): number of generated captcha images
        img_width(int): width of generated captcha images
        img_height(int): height of generated captcha images
        max_digits(int): max number of digits in each captcha images. For each captcha images, number of digits is in
        range [1,max_digits]
        process_num(int): number of process to generate captcha images, default is 16
    """
    cur_script_path = os.path.dirname(os.path.realpath(__file__))
    path_data = os.path.join(cur_script_path, "data")
    if not os.path.exists(path_data):
        os.mkdir(path_data)
    path = os.path.join(path_data, name)
    print("Generating dataset [{}] under {}...".format(name, path))
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)
    img_num_per_thread = m.ceil(img_num / process_num)

    processes = []
    for i in range(process_num):
        start = i * img_num_per_thread
        end = start + img_num_per_thread if i != (process_num - 1) else img_num
        p = Process(target=_generate_captcha_per_process,
                    args=(path, img_num, start, end, img_width, img_height, max_digits))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print("Generating dataset [{}] finished, total number is {}!".format(name, img_num))


if __name__ == '__main__':
    generate_captcha("test", img_num=10000, img_width=160, img_height=64, max_digits=4)
    generate_captcha("train", img_num=50000, img_width=160, img_height=64, max_digits=4)
