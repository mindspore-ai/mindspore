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
import argparse
import matplotlib
import numpy as np
import cv2

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

def statistic_normalize_img(img, statistic_norm=True):
    """Statistic normalize images."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    if statistic_norm:
        img = (img - mean) / std
    img = img.astype(np.float32)
    return img

def preprocess(im_crops):
    """
    TODO:
        1. to float with scale from 0 to 1
        2. resize to (64, 128) as Market1501 dataset did
        3. concatenate to a numpy array
        3. to torch Tensor
        4. normalize
    """
    def _resize(im, size):
        return cv2.resize(im.astype(np.float32)/255., size)
    im_batch = []
    size = (64, 128)
    for im in im_crops:
        im = _resize(im, size)
        im = statistic_normalize_img(im)
        im = im.transpose(2, 0, 1).copy()
        im = np.expand_dims(im, 0)
        im_batch.append(im)

    im_batch = np.array(im_batch)
    return im_batch


def get_features(bbox_xywh, ori_img):
    im_crops = []
    for box in bbox_xywh:
        im = extract_image_patch(ori_img, box)
        if im is None:
            print("WARNING: Failed to extract image patch: %s." % str(box))
            im = np.random.uniform(
                0., 255., ori_img.shape).astype(np.uint8)
        im_crops.append(im)
    if im_crops:
        features = preprocess(im_crops)
    else:
        features = np.array([])
    return features

def generate_detections(mot_dir, img_path, det_path=None):
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

    count = 0
    for sequence in os.listdir(mot_dir):
        print("Processing %s" % sequence)
        sequence_dir = os.path.join(mot_dir, sequence)

        image_dir = os.path.join(sequence_dir, "img1")
        #image_dir = os.path.join(mot_dir, "img1")
        image_filenames = {
            int(os.path.splitext(f)[0]): os.path.join(image_dir, f)
            for f in os.listdir(image_dir)}

        if det_path is not None:
            detection_dir = os.path.join(det_path, sequence)
        else:
            detection_dir = os.path.join(sequence_dir, sequence)
        detection_file = os.path.join(detection_dir, "det/det.txt")
        detections_in = np.loadtxt(detection_file, delimiter=',')

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
            bgr_image = cv2.imread(image_filenames[frame_idx], cv2.IMREAD_COLOR)
            features = get_features(rows[:, 2:6].copy(), bgr_image)

            for data in features:
                file_name = "DeepSort_data_bs" + str(1) + "_" + str(count) + ".bin"
                file_path = img_path + "/" + file_name
                data.tofile(file_path)
                count += 1

def parse_args():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Ascend 310 feature extractor")
    parser.add_argument('--data_path', type=str, default='', help='MOT directory.')
    parser.add_argument('--det_path', type=str, default='', help='Det directory.')
    parser.add_argument('--result_path', type=str, default='', help='Inference output directory.')

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    image_path = os.path.join(args.result_path, "00_data")
    os.mkdir(image_path)
    generate_detections(args.data_path, image_path, args.det_path)
