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
import errno
import argparse
import ast
import matplotlib

import numpy as np
import cv2

from mindspore.train.model import ParallelMode
from mindspore.communication.management import init
from mindspore import context
from src.deep.feature_extractor import Extractor

matplotlib.use("Agg")
ASCEND_SLOG_PRINT_TO_STDOUT = 1


def extract_image_patch(image, bbox, patch_shape=None):
    """Extract image patch from bounding box.

    Parameters
    ----------
    image : ndarray
        The full image.
    bbox : array_like
        The bounding box in format (x, y, width, height).
    patch_shape : Optional[array_like]
        This parameter can be used to enforce a desired patch shape
        (height, width). First, the `bbox` is adapted to the aspect ratio
        of the patch shape, then it is clipped at the image boundaries.
        If None, the shape is computed from :arg:`bbox`.

    Returns
    -------
    ndarray | NoneType
        An image patch showing the :arg:`bbox`, optionally reshaped to
        :arg:`patch_shape`.
        Returns None if the bounding box is empty or fully outside of the image
        boundaries.

    """
    bbox = np.array(bbox)
    if patch_shape is not None:
        # correct aspect ratio to patch shape
        target_aspect = float(patch_shape[1]) / patch_shape[0]
        new_width = target_aspect * bbox[3]
        bbox[0] -= (new_width - bbox[2]) / 2
        bbox[2] = new_width

    # convert to top left, bottom right
    bbox[2:] += bbox[:2]
    bbox = bbox.astype(np.int)

    # clip at image boundaries
    bbox[:2] = np.maximum(0, bbox[:2])
    bbox[2:] = np.minimum(np.asarray(image.shape[:2][::-1]) - 1, bbox[2:])
    if np.any(bbox[:2] >= bbox[2:]):
        return None
    sx, sy, ex, ey = bbox
    image = image[sy:ey, sx:ex]
    return image


class ImageEncoder:

    def __init__(self, model_path, batch_size=32):

        self.extractor = Extractor(model_path, batch_size)

    def _get_features(self, bbox_xywh, ori_img):
        im_crops = []
        self.height, self.width = ori_img.shape[:2]
        for box in bbox_xywh:
            im = extract_image_patch(ori_img, box)
            if im is None:
                print("WARNING: Failed to extract image patch: %s." % str(box))
                im = np.random.uniform(
                    0., 255., ori_img.shape).astype(np.uint8)
            im_crops.append(im)
        if im_crops:
            features = self.extractor(im_crops)
        else:
            features = np.array([])
        return features


    def __call__(self, image, boxes, batch_size=32):
        features = self._get_features(boxes, image)
        return features


def create_box_encoder(model_filename, batch_size=32):
    image_encoder = ImageEncoder(model_filename, batch_size)

    def encoder_box(image, boxes):
        return image_encoder(image, boxes)

    return encoder_box


def generate_detections(encoder_boxes, mot_dir, output_dir, det_path=None, detection_dir=None):
    """Generate detections with features.

    Parameters
    ----------
    encoder : Callable[image, ndarray] -> ndarray
        The encoder function takes as input a BGR color image and a matrix of
        bounding boxes in format `(x, y, w, h)` and returns a matrix of
        corresponding feature vectors.
    mot_dir : str
        Path to the MOTChallenge directory (can be either train or test).
    output_dir
        Path to the output directory. Will be created if it does not exist.
    detection_dir
        Path to custom detections. The directory structure should be the default
        MOTChallenge structure: `[sequence]/det/det.txt`. If None, uses the
        standard MOTChallenge detections.

    """
    if detection_dir is None:
        detection_dir = mot_dir
    try:
        os.makedirs(output_dir)
    except OSError as exception:
        if exception.errno == errno.EEXIST and os.path.isdir(output_dir):
            pass
        else:
            raise ValueError(
                "Failed to created output directory '%s'" % output_dir)
    for sequence in os.listdir(mot_dir):
        print("Processing %s" % sequence)
        sequence_dir = os.path.join(mot_dir, sequence)

        image_dir = os.path.join(sequence_dir, "img1")
        #image_dir = os.path.join(mot_dir, "img1")
        image_filenames = {
            int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
            for f in os.listdir(image_dir)}
        if det_path:
            detection_dir = os.path.join(det_path, sequence)
        else:
            detection_dir = os.path.join(sequence_dir, sequence)
        detection_file = os.path.join(detection_dir, "det/det.txt")

        detections_in = np.loadtxt(detection_file, delimiter=',')
        detections_out = []

        frame_indices = detections_in[:, 0].astype(np.int)
        min_frame_idx = frame_indices.astype(np.int).min()
        max_frame_idx = frame_indices.astype(np.int).max()
        for frame_idx in range(min_frame_idx, max_frame_idx + 1):
            print("Frame %05d/%05d" % (frame_idx, max_frame_idx))
            mask = frame_indices == frame_idx
            rows = detections_in[mask]

            if frame_idx not in image_filenames:
                print("WARNING could not find image for frame %d" % frame_idx)
                continue
            bgr_image = cv2.imread(
                image_filenames[frame_idx], cv2.IMREAD_COLOR)
            features = encoder_boxes(bgr_image, rows[:, 2:6].copy())
            detections_out += [np.r_[(row, feature)]
                               for row, feature in zip(rows, features)]

        output_filename = os.path.join(output_dir, "%s.npy" % sequence)
        print(output_filename)
        np.save(
            output_filename, np.asarray(detections_out), allow_pickle=False)


def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Re-ID feature extractor")
    parser.add_argument('--run_distribute', type=ast.literal_eval,
                        default=False, help='Run distribute')
    parser.add_argument('--run_modelarts', type=ast.literal_eval,
                        default=False, help='Run distribute')
    parser.add_argument("--device_id", type=int, default=4,
                        help="Use which device.")
    parser.add_argument('--data_url', type=str,
                        default='', help='Det directory.')
    parser.add_argument('--train_url', type=str, default='',
                        help='Train output directory.')
    parser.add_argument('--det_url', type=str, default='',
                        help='Train output directory.')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='Batach size.')
    parser.add_argument("--ckpt_url", type=str, default='',
                        help="Path to checkpoint.")
    parser.add_argument("--model_name", type=str,
                        default="deepsort-30000_24.ckpt", help="Name of checkpoint.")
    parser.add_argument(
        "--detection_dir", help="Path to custom detections. Defaults to", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    context.set_context(mode=context.GRAPH_MODE, enable_auto_mixed_precision=True,
                        device_target="Ascend", save_graphs=False)
    args = parse_args()
    if args.run_modelarts:
        import moxing as mox
        device_id = int(os.getenv('DEVICE_ID'))
        device_num = int(os.getenv('RANK_SIZE'))
        context.set_context(device_id=device_id)
        local_data_url = '/cache/data'
        local_ckpt_url = '/cache/ckpt'
        local_train_url = '/cache/train'
        local_det_url = '/cache/det'
        mox.file.copy_parallel(args.ckpt_url, local_ckpt_url)
        mox.file.copy_parallel(args.data_url, local_data_url)
        mox.file.copy_parallel(args.det_url, local_det_url)
        if device_num > 1:
            init()
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
        DATA_DIR = local_data_url + '/'
        ckpt_dir = local_ckpt_url + '/'
        det_dir = local_det_url + '/'
    else:
        if args.run_distribute:
            device_id = int(os.getenv('DEVICE_ID'))
            device_num = int(os.getenv('RANK_SIZE'))
            context.set_context(device_id=device_id)
            init()
            context.reset_auto_parallel_context()
            context.set_auto_parallel_context(device_num=device_num,
                                              parallel_mode=ParallelMode.DATA_PARALLEL, gradients_mean=True)
        else:
            context.set_context(device_id=args.device_id)
            device_num = 1
            device_id = args.device_id
        DATA_DIR = args.data_url
        local_train_url = args.train_url
        ckpt_dir = args.ckpt_url
        det_dir = args.det_url

    encoder = create_box_encoder(
        ckpt_dir+args.model_name, batch_size=args.batch_size)
    generate_detections(encoder, DATA_DIR, local_train_url, det_path=det_dir)
    if args.run_modelarts:
        mox.file.copy_parallel(src_url=local_train_url, dst_url=args.train_url)
