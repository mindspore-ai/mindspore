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
"""convert tsv file to images"""
import base64
import struct
import os
import argparse

parser = argparse.ArgumentParser(description='convert tsv file to images')
parser.add_argument('--file_path', default='./FaceImageCroppedWithAlignment.tsv', type=str,
                    help='the path of csv file')
parser.add_argument('--output_dir', default='./FaceImageCroppedWithAlignment/', type=str,
                    help='the path of converted images')
args = parser.parse_args()


def read_line(line):
    """read line"""
    m_id, image_search_rank, image_url, page_url, face_id, face_rectangle, face_data = line.split("\t")
    rect = struct.unpack("ffff", base64.b64decode(face_rectangle))
    result = {
        'm_id': m_id,
        'image_search_rank': image_search_rank,
        'image_url': image_url,
        'page_url': page_url,
        'face_id': face_id,
        'rect': rect,
        'face_data': base64.b64decode(face_data)
    }
    return result


def write_image(filename, data):
    """write image"""
    with open(filename, "wb") as f:
        f.write(data)


def unpack(file_name, output_dir):
    """unpack file"""
    i = 0
    with open(file_name, "r", encoding="utf-8") as f:
        for line in f:
            result = read_line(line)
            img_dir = os.path.join(output_dir, result['m_id'])
            if not os.path.exists(img_dir):
                os.mkdir(img_dir)
            img_name = "%s-%s" % (result['image_search_rank'], result['face_id']) + ".jpg"
            write_image(os.path.join(img_dir, img_name), result['face_data'])
            i += 1
            if i % 1000 == 0:
                print(i, "images finished")
        print("all finished")


def main(file_name, output_dir):
    """main function"""
    unpack(file_name, output_dir)


if __name__ == '__main__':
    main(file_name=args.file_path, output_dir=args.output_dir)
