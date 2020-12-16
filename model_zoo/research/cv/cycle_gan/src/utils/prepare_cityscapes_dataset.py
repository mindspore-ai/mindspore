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
"""prepare cityscapes dataset to cyclegan format"""
import os
import argparse
import glob
from PIL import Image

parser = argparse.ArgumentParser()
parser.add_argument('--gtFine_dir', type=str, required=True,
                    help='Path to the Cityscapes gtFine directory.')
parser.add_argument('--leftImg8bit_dir', type=str, required=True,
                    help='Path to the Cityscapes leftImg8bit_trainvaltest directory.')
parser.add_argument('--output_dir', type=str, required=True,
                    default='./cityscapes',
                    help='Directory the output images will be written to.')
opt = parser.parse_args()

def load_resized_img(path):
    """Load image with RGB and resize to (256, 256)"""
    return Image.open(path).convert('RGB').resize((256, 256))

def check_matching_pair(segmap_path, photo_path):
    """Check the segment images and photo images are matched or not."""
    segmap_identifier = os.path.basename(segmap_path).replace('_gtFine_color', '')
    photo_identifier = os.path.basename(photo_path).replace('_leftImg8bit', '')

    assert segmap_identifier == photo_identifier, \
        f"[{segmap_path}] and [{photo_path}] don't seem to be matching. Aborting."


def process_cityscapes(gtFine_dir, leftImg8bit_dir, output_dir, phase):
    """Process citycapes dataset to cyclegan dataset format."""
    save_phase = 'test' if phase == 'val' else 'train'
    savedir = os.path.join(output_dir, save_phase)
    os.makedirs(savedir + 'A', exist_ok=True)
    os.makedirs(savedir + 'B', exist_ok=True)
    print(f"Directory structure prepared at {output_dir}")

    segmap_expr = os.path.join(gtFine_dir, phase) + "/*/*_color.png"
    segmap_paths = glob.glob(segmap_expr)
    segmap_paths = sorted(segmap_paths)

    photo_expr = os.path.join(leftImg8bit_dir, phase) + "/*/*_leftImg8bit.png"
    photo_paths = glob.glob(photo_expr)
    photo_paths = sorted(photo_paths)

    assert len(segmap_paths) == len(photo_paths), \
        "{} images that match [{}], and {} images that match [{}]. Aborting.".format(
            len(segmap_paths), segmap_expr, len(photo_paths), photo_expr)

    for i, (segmap_path, photo_path) in enumerate(zip(segmap_paths, photo_paths)):
        check_matching_pair(segmap_path, photo_path)
        segmap = load_resized_img(segmap_path)
        photo = load_resized_img(photo_path)

        # data for cyclegan where the two images are stored at two distinct directories
        savepath = os.path.join(savedir + 'A', f"{i + 1}.jpg")
        photo.save(savepath)
        savepath = os.path.join(savedir + 'B', f"{i + 1}.jpg")
        segmap.save(savepath)

        if i % (len(segmap_paths) // 10) == 0:
            print("%d / %d: last image saved at %s, " % (i, len(segmap_paths), savepath))


if __name__ == '__main__':
    print('Preparing Cityscapes Dataset for val phase')
    process_cityscapes(opt.gtFine_dir, opt.leftImg8bit_dir, opt.output_dir, "val")
    print('Preparing Cityscapes Dataset for train phase')
    process_cityscapes(opt.gtFine_dir, opt.leftImg8bit_dir, opt.output_dir, "train")
    print('Done')
