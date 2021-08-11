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


import argparse
import json
import logging
import os

import matplotlib as mpl
import matplotlib.colors as mplc
import matplotlib.figure as mplf
import matplotlib.image as mpi
import numpy as np

# Part of the code reference https://github.com/facebookresearch/detectron2/tree/v0.2.1

# fmt: off
# RGB:
_COLORS = np.array([
    0.000, 0.447, 0.741, 0.850, 0.325, 0.098, 0.929, 0.694, 0.125, 0.494,
    0.184, 0.556, 0.466, 0.674, 0.188, 0.301, 0.745, 0.933, 0.635, 0.078,
    0.184, 0.300, 0.300, 0.300, 0.600, 0.600, 0.600, 1.000, 0.000, 0.000,
    1.000, 0.500, 0.000, 0.749, 0.749, 0.000, 0.000, 1.000, 0.000, 0.000,
    0.000, 1.000, 0.667, 0.000, 1.000, 0.333, 0.333, 0.000, 0.333, 0.667,
    0.000, 0.333, 1.000, 0.000, 0.667, 0.333, 0.000, 0.667, 0.667, 0.000,
    0.667, 1.000, 0.000, 1.000, 0.333, 0.000, 1.000, 0.667, 0.000, 1.000,
    1.000, 0.000, 0.000, 0.333, 0.500, 0.000, 0.667, 0.500, 0.000, 1.000,
    0.500, 0.333, 0.000, 0.500, 0.333, 0.333, 0.500, 0.333, 0.667, 0.500,
    0.333, 1.000, 0.500, 0.667, 0.000, 0.500, 0.667, 0.333, 0.500, 0.667,
    0.667, 0.500, 0.667, 1.000, 0.500, 1.000, 0.000, 0.500, 1.000, 0.333,
    0.500, 1.000, 0.667, 0.500, 1.000, 1.000, 0.500, 0.000, 0.333, 1.000,
    0.000, 0.667, 1.000, 0.000, 1.000, 1.000, 0.333, 0.000, 1.000, 0.333,
    0.333, 1.000, 0.333, 0.667, 1.000, 0.333, 1.000, 1.000, 0.667, 0.000,
    1.000, 0.667, 0.333, 1.000, 0.667, 0.667, 1.000, 0.667, 1.000, 1.000,
    1.000, 0.000, 1.000, 1.000, 0.333, 1.000, 1.000, 0.667, 1.000, 0.333,
    0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000,
    0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000,
    0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000,
    1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000, 0.333, 0.000, 0.000,
    0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833, 0.000, 0.000, 1.000,
    0.000, 0.000, 0.000, 0.143, 0.143, 0.143, 0.857, 0.857, 0.857, 1.000,
    1.000, 1.000
]).astype(np.float32).reshape(-1, 3)


# fmt: on
def random_color(rgb=False, maximum=255):
    """
    Args:
        rgb (bool): whether to return RGB colors or BGR colors.
        maximum (int): either 255 or 1

    Returns:
        ndarray: a vector of 3 numbers
    """
    idx = np.random.randint(0, len(_COLORS))
    ret = _COLORS[idx] * maximum
    if not rgb:
        ret = ret[::-1]
    return ret


def draw_bbox_and_label(ax, boxes, default_font_size):
    # draw bbox and label
    color = 'g'
    for box in boxes:
        start_pos = (box["x0"], box["y0"])
        ax.add_patch(
            mpl.patches.Rectangle(
                start_pos,
                box["width"],
                box["height"],
                fill=False,
                edgecolor=color,
                linewidth=max(default_font_size / 4, 1),
                alpha=0.5,
                linestyle="-",
            ))

        color = np.maximum(list(mplc.to_rgb(color)), 0.2)
        color[np.argmax(color)] = max(0.8, np.max(color))
        ax.text(
            *start_pos,
            "{:s} {:.2%}".format(box["label"], box["confidence"]),
            size=default_font_size,
            family="sans-serif",
            bbox={
                "facecolor": "black",
                "alpha": 0.8,
                "pad": 0.7,
                "edgecolor": "none"
            },
            verticalalignment="top",
            horizontalalignment="left",
            color=color,
            zorder=10,
            rotation=0,
        )


def draw_box(boxes, img_path, label_img_dir):
    img = mpi.imread(img_path)
    boxes = np.array(boxes)
    img_height, img_width = img.shape[0], img.shape[1]
    fig = mplf.Figure((img_width / 100, img_height / 100), frameon=False)
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.axis("off")
    ax.imshow(img,
              extent=(0, img_width, img_height, 0),
              interpolation="nearest")

    default_font_size = max(np.sqrt(img_width * img_height) // 90, 10)
    draw_bbox_and_label(ax, boxes, default_font_size)

    _, file_name = os.path.split(img_path)
    label_path = os.path.join(label_img_dir, file_name)
    fig.savefig(label_path)


def draw_label(res_file, img_path, label_img_dir):
    with open(res_file, "r") as fp:
        result = json.loads(fp.read())
    if not result:
        logging.error("The result data is error, img_path(%s).", img_path)
        return

    bound_boxes = list()
    data = result.get("MxpiObject")
    if not data:
        logging.error("The result data is error, img_path(%s).", img_path)
        return

    for bbox in data:
        class_vec = bbox.get("classVec")[0]
        bound_box = dict(
            label=class_vec.get("className"),
            confidence=class_vec.get("confidence"),
            x0=int(bbox["x0"]),
            y0=int(bbox["y0"]),
            x1=int(bbox["x1"]),
            y1=int(bbox["y1"]),
            width=int(bbox["x1"] - bbox["x0"]),
            height=int(bbox["y1"] - bbox["y0"])
        )

        bound_boxes.append(bound_box)
    draw_box(bound_boxes, img_path, label_img_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draw inference result on the image.")
    parser.add_argument('--image', type=str, required=True, help="The origin image.")
    parser.add_argument('--pipeline', type=str, required=True, help="The pipeline file.")
    args_opt = parser.parse_args()
    res_tmp_file = "predict_result.json"
    draw_label(res_tmp_file, args_opt.image, './img_label')
