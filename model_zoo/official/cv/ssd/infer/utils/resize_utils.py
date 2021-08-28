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

import os
import cv2


def resize_one_img(org_img, resize_to=(608, 608), interpolation=cv2.INTER_LINEAR, output_file_name=None):
    if not output_file_name:
        output_file_name = "_boxed".join(os.path.splitext(org_img))
    else:
        os.makedirs(os.path.dirname(output_file_name), exist_ok=True)

    org_img_obj = cv2.imread(org_img, cv2.IMREAD_UNCHANGED)
    resized_img = cv2.resize(
        org_img_obj, resize_to, interpolation=interpolation
    )
    cv2.imwrite(output_file_name, resized_img)


def resize_imgs(img_dir, output_dir, resize_to=(608, 608), interpolation=cv2.INTER_LINEAR):
    imgs = os.listdir(img_dir)
    img_full_names = [
        os.path.join(img_dir, img)
        for img in imgs
        if img.endswith(".jpg") and "boxed" not in img
    ]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print("Begin resize all images...")
    for img_name in img_full_names:
        resize_one_img(
            img_name,
            resize_to,
            interpolation,
            output_file_name=os.path.join(output_dir, os.path.basename(img_name)),
        )
    print(f"Resize all: {len(img_full_names)} images finished.")
