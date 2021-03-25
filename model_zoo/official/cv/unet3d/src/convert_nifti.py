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
from pathlib import Path
import SimpleITK as sitk
from src.config import config

parser = argparse.ArgumentParser()
parser.add_argument("--input_path", type=str, help="Input image directory to be processed.")
parser.add_argument("--output_path", type=str, help="Output file path.")
args = parser.parse_args()

def get_list_of_files_in_dir(directory, file_types='*'):
    """
    Get list of certain format files.

    Args:
        directory (str): The input directory for image.
        file_types (str): The file_types to filter the files.
    """
    return [f for f in Path(directory).glob(file_types) if f.is_file()]

def convert_nifti(input_dir, output_dir, roi_size, file_types):
    """
    Convert dataset into mifti format.

    Args:
        input_dir (str): The input directory for image.
        output_dir (str): The output directory to save nifti format data.
        roi_size (str): The size to crop the image.
        file_types: File types to convert into nifti.
    """
    file_list = get_list_of_files_in_dir(input_dir, file_types)
    for file_name in file_list:
        file_name = str(file_name)
        input_file_name, _ = os.path.splitext(os.path.basename(file_name))
        img = sitk.ReadImage(file_name)
        image_array = sitk.GetArrayFromImage(img)
        D, H, W = image_array.shape
        if H < roi_size[0] or W < roi_size[1] or D < roi_size[2]:
            print("file {} size is smaller than roi size, ignore it.".format(input_file_name))
            continue
        output_path = os.path.join(output_dir, input_file_name + ".nii.gz")
        sitk.WriteImage(img, output_path)
        print("create output file {} success.".format(output_path))

if __name__ == '__main__':
    convert_nifti(args.input_path, args.output_path, config.roi_size, "*.mhd")
