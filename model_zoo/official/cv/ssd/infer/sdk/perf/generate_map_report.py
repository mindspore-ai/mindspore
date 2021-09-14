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
from datetime import datetime

from absl import flags
from absl import app
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

PRINT_LINES_TEMPLATE = """
Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = %.3f
Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = %.3f
Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = %.3f
Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = %.3f
Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = %.3f
Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = %.3f
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = %.3f
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = %.3f
Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = %.3f
Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = %.3f
Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = %.3f
Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = %.3f
"""

FLAGS = flags.FLAGS
flags.DEFINE_string(
    name="annotations_json",
    default=None,
    help="annotations_json file path name",
)

flags.DEFINE_string(
    name="det_result_json", default=None, help="det_result json file"
)

flags.DEFINE_enum(
    name="anno_type",
    default="bbox",
    enum_values=["segm", "bbox", "keypoints"],
    help="Annotation type",
)

flags.DEFINE_string(
    name="output_path_name",
    default=None,
    help="Where to out put the result files.",
)

flags.mark_flag_as_required("annotations_json")
flags.mark_flag_as_required("det_result_json")
flags.mark_flag_as_required("output_path_name")


def main(unused_arg):
    del unused_arg
    out_put_dir = os.path.dirname(FLAGS.output_path_name)
    if not os.path.exists(out_put_dir):
        os.makedirs(out_put_dir)

    fw = open(FLAGS.output_path_name, "a+")
    now_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    head_info = f"{'-'*50}mAP Test starts @ {now_time_str}{'-'*50}\n"
    fw.write(head_info)
    fw.flush()

    cocoGt = COCO(FLAGS.annotations_json)
    cocoDt = cocoGt.loadRes(FLAGS.det_result_json)
    cocoEval = COCOeval(cocoGt, cocoDt, FLAGS.anno_type)
    cocoEval.params.imgIds = sorted(cocoGt.getImgIds())
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

    format_lines = [
        line for line in PRINT_LINES_TEMPLATE.splitlines() if line.strip()
    ]
    for i, line in enumerate(format_lines):
        fw.write(line % cocoEval.stats[i] + "\n")

    end_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tail_info = f"{'-'*50}mAP Test ends @ {end_time_str}{'-'*50}\n"
    fw.write(tail_info)
    fw.close()


if __name__ == "__main__":
    app.run(main)
