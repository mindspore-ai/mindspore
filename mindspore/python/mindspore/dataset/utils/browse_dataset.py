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
# ==============================================================================
"""Visualization for detection/segmentation dataset.
"""
import os
import sys
import importlib
import numpy as np

from mindspore import log as logger


def imshow_det_bbox(image, bboxes, labels, segm=None, class_names=None, score_threshold=0, bbox_color=(0, 255, 0),
                    text_color=(203, 192, 255), mask_color=(128, 0, 128), thickness=2, font_size=0.8, show=True,
                    win_name="win", wait_time=2000, out_file=None):
    """Draw an image with given bboxes and class labels (with scores).

    Args:
        image (numpy.ndarray): The image to be displayed, shaped :math:`(C, H, W)` or :math:`(H, W, C)`,
            formatted RGB.
        bboxes (numpy.ndarray): Bounding boxes (with scores), shaped :math:`(N, 4)` or :math:`(N, 5)`,
            data should be ordered with (N, x, y, w, h).
        labels (numpy.ndarray): Labels of bboxes, shaped :math:`(N, 1)`.
        segm (numpy.ndarray): The segmentation masks of image in M classes, shaped :math:`(M, H, W)`.
            Default: ``None``.
        class_names (list[str], tuple[str], dict): Names of each class to map label to class name.
            Default: ``None``, only display label.
        score_threshold (float): Minimum score of bboxes to be shown. Default: ``0``.
        bbox_color (tuple(int)): Color of bbox lines.
            The tuple of color should be in BGR order. Default: ``(0, 255 ,0)``, means 'green'.
        text_color (tuple(int)): Color of texts.
            The tuple of color should be in BGR order. Default: ``(203, 192, 255)``, means 'pink'.
        mask_color (tuple(int)): Color of mask.
            The tuple of color should be in BGR order. Default: ``(128, 0, 128)``, means 'purple'.
        thickness (int): Thickness of lines. Default: ``2``.
        font_size (int, float): Font size of texts. Default: ``0.8``.
        show (bool): Whether to show the image. Default: ``True``.
        win_name (str): The window name. Default: ``"win"``.
        wait_time (int): Waiting time delay for a key event in milliseconds. During image display,
            if there is no key pressed, wait for this time then jump to next image. If ESC is pressed,
            quit display immediately. If any other key is pressed, stop waiting then jump to
            next image directly. Default: ``2000`` , wait 2000ms then jump to next image.
        out_file (str, optional): The filename to write the imagee. Default: ``None``. File extension name
            is required to indicate the image compression type, e.g. 'jpg', 'png'.

    Returns:
        ndarray, The image with bboxes drawn on it.

    Note:
        This interface relies on the `opencv-python` library.

    Raises:
        ImportError: If `opencv-python` is not installed.
        AssertionError: If `image` is not in (H, W, C) or (C, H, W) format.
        AssertionError: If `bboxes` is not in (N, 4) or (N, 5) format.
        AssertionError: If `labels` is not in (N, 1) format.
        AssertionError: If `segm` is not in (M, H, W) format.
        AssertionError: If `class_names` is not of type list, tuple or dict.
        AssertionError: If `bbox_color` is not a tuple in format of (B, G, R).
        AssertionError: If `text_color` is not a tuple in format of (B, G, R).
        AssertionError: If `mask_color` is not a tuple in format of (B, G, R).

    Examples:
        >>> import numpy as np
        >>> import mindspore.dataset as ds
        >>> from mindspore.dataset.utils import imshow_det_bbox
        >>>
        >>> # Read Detection dataset, such as VOC2012.
        >>> voc_dataset_dir = "/path/to/voc_dataset_directory"
        >>> dataset = ds.VOCDataset(voc_dataset_dir, task="Detection", shuffle=False, decode=True, num_samples=5)
        >>> dataset_iter = dataset.create_dict_iterator(output_numpy=True, num_epochs=1)
        >>>
        >>> # draw dataset
        >>> for index, data in enumerate(dataset_iter):
        ...     image = data["image"]
        ...     bbox = data["bbox"]
        ...     label = data["label"]
        ...     # draw image with bboxes
        ...     imshow_det_bbox(image, bbox, label,
        ...                     class_names=['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat',
        ...                                  'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person',
        ...                                  'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'],
        ...                     win_name="my_window",
        ...                     wait_time=5000,
        ...                     show=True,
        ...                     out_file="voc_dataset_{}.jpg".format(str(index)))

    Examples using `imshow_det_bbox` on VOC2012:

    .. image:: browse_dataset.png

    """

    try:
        cv2 = importlib.import_module("cv2")
    except ModuleNotFoundError:
        raise ImportError("Importing cv2 failed, try to install it by running `pip install opencv-python`.")

    # validation
    assert isinstance(image, np.ndarray) and image.ndim == 3 and (image.shape[0] == 3 or image.shape[2] == 3), \
        "image must be a ndarray in (H, W, C) or (C, H, W) format."
    if bboxes is not None:
        assert isinstance(bboxes, np.ndarray) and bboxes.ndim == 2 and (bboxes.shape[1] == 4 or bboxes.shape[1] == 5), \
            "bboxes must be a ndarray in (N, 4) or (N, 5) format."
        assert isinstance(labels, np.ndarray) and labels.ndim == 2 and labels.shape[1] == 1 and \
               labels.shape[0] == bboxes.shape[
                   0], "labels must be a ndarray in (N, 1) format and has same N with bboxes."
    if segm is not None:
        assert isinstance(segm, np.ndarray) and segm.ndim == 3, "segm must be a ndarray in (M, H, W) format."
        H, W = (image.shape[0], image.shape[1]) if image.shape[2] == 3 else (image.shape[1], image.shape[2])
        assert H == segm.shape[1] and W == segm.shape[2], "segm must has same height and width with image."
        if bboxes is not None:
            assert bboxes.shape[0] <= segm.shape[0], "number of segm masks must not be less than the number of bboxes."
    assert isinstance(class_names, (tuple, list, dict)), "class_names must be a list, tuple or dict."
    assert isinstance(bbox_color, tuple) and len(bbox_color) == 3, \
        "bbox_color must be a three tuple, formatted (B, G, R)."
    assert isinstance(text_color, tuple) and len(text_color) == 3, \
        "text_color must be a three tuple, formatted (B, G, R)."
    assert isinstance(mask_color, tuple) and len(mask_color) == 3, \
        "mask_color must be a three tuple, formatted (B, G, R)."
    assert isinstance(thickness, int), "thickness must be an int."
    assert thickness >= 0, "thickness must be larger than or equal to zero."
    assert isinstance(font_size, (int, float)), "font_size must be an int or float."
    assert font_size >= 0, "font_size must be larger than or equal to zero."
    assert isinstance(show, bool), "show must be a bool."
    assert isinstance(win_name, str), "win_name must be a str."
    assert isinstance(wait_time, int), "wait_time must be an int."
    assert wait_time >= 0, "wait_time must be larger than or equal to zero."
    if out_file is not None:
        assert isinstance(out_file, str), "out_file must be a str."

    if score_threshold > 0:
        assert bboxes.shape[1] == 5
    if not show:
        assert out_file is not None

    # image
    if image.shape[0] == 3:
        image = image.transpose((1, 2, 0))
    draw_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if bboxes is not None:
        bbox_num = bboxes.shape[0]
        for i in range(bbox_num):
            draw_bbox = bboxes[i]
            if len(draw_bbox) > 4:
                if draw_bbox[4] < score_threshold:
                    continue
            # bbox
            x1, y1 = int(draw_bbox[0]), int(draw_bbox[1])
            x2, y2 = int(draw_bbox[0] + draw_bbox[2]), int(draw_bbox[1] + draw_bbox[3])
            cv2.rectangle(draw_image, (x1, y1), (x2, y2), bbox_color, thickness)
            # label
            try:
                draw_label = str(class_names[labels[i][0]]) if class_names is not None else f'class {labels[i][0]}'
            except (IndexError, KeyError):
                draw_label = f'class {labels[i][0]}'
            if len(draw_bbox) > 4:
                draw_label += f'|{draw_bbox[-1]:.02f}'
            cv2.putText(draw_image, draw_label, (x1, y2), cv2.FONT_HERSHEY_SIMPLEX, font_size, text_color, thickness)
            if segm is not None:
                mask = segm[i].astype(bool)
                draw_image[mask] = draw_image[mask] * 0.5 + np.array(mask_color) * 0.5
    else:
        if segm is not None:
            segm_num = segm.shape[0]
            for i in range(segm_num):
                mask = segm[i].astype(bool)
                draw_image[mask] = draw_image[mask] * 0.5 + np.array(mask_color) * 0.5
    if show:
        cv2.imshow(win_name, draw_image)
        if cv2.waitKey(wait_time) == 27:
            sys.exit()
    if out_file:
        logger.info("Saving image file with name: " + out_file + "...")
        cv2.imwrite(out_file, draw_image)
        os.chmod(out_file, 0o600)
    return draw_image
