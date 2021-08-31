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

import json
import os

import cv2


def restore_coordinates(resized_json_det_file, img_dir, resized_to=(640, 640)):
    img_path_names = [
        os.path.join(img_dir, name) for name in os.listdir(img_dir)
    ]
    resize_dict = dict()
    for img_path_name in img_path_names:
        img_np = cv2.imread(img_path_name, cv2.IMREAD_UNCHANGED)
        shape = img_np.shape
        old_height, old_width, = shape[0], shape[1]
        x_ratio = old_width * 1.0 / resized_to[0]
        y_ratio = old_height * 1.0 / resized_to[1]
        img_id = int(os.path.basename(img_path_name).split(".")[0])
        resize_dict[img_id] = (x_ratio, y_ratio)

    old_json_det_dict_list = []
    resized_json_det_dict_list = json.load(open(resized_json_det_file))
    for resized_json_det_dict in resized_json_det_dict_list:
        img_id = resized_json_det_dict.get("image_id")
        nx, ny, nw, nh = resized_json_det_dict.get("bbox")
        x_ratio, y_ratio = resize_dict.get(img_id)
        old_bbox = [
            round(nx * x_ratio, 2),
            round(ny * y_ratio, 2),
            round(nw * x_ratio, 2),
            round(nh * y_ratio, 2),
        ]
        old_json_det_dict_list.append(
            dict(
                image_id=img_id,
                bbox=old_bbox[:],
                category_id=resized_json_det_dict.get("category_id"),
                score=resized_json_det_dict.get("score"),
            )
        )

    return old_json_det_dict_list


if __name__ == "__main__":
    result_json = "./origin_pb_det_resized_result/resized_origin_coco_det_result.json"
    restore_json_path = "./restored_json_det/coordinates_restored_det.json"
    real_label_dir = "./cocoapi/cocoapi-master/val2017"
    resize_height = 608
    resize_width = 608
    old_json_det_dit_list = restore_coordinates(
        result_json,
        real_label_dir,
        resized_to=(resize_height, resize_width),
    )
    restored_json = (
        restore_json_path
    )
    with open(restored_json, "w") as f:
        f.write(json.dumps(old_json_det_dit_list))
