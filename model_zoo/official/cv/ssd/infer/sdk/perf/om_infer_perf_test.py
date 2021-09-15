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
import threading
import time
from datetime import datetime

import MxpiDataType_pb2 as MxpiDataType
import cv2
from StreamManagerApi import InProtobufVector
from StreamManagerApi import MxDataInput
from StreamManagerApi import MxProtobufIn
from StreamManagerApi import StreamManagerApi
from StreamManagerApi import StringVector
from absl import app
from absl import flags

BOXED_IMG_DIR = None
TXT_DIR = None
PERF_REPORT_TXT = None
DET_RESULT_RESIZED_JSON = None
DET_RESULT_JSON = None

FLAGS = flags.FLAGS
infer_ret_list_lock = threading.Lock()
det_restore_ratio = dict()

flags.DEFINE_string(
    name="img_dir", default=None, help="Directory of images to infer"
)

flags.DEFINE_string(
    name="pipeline_config",
    default=None,
    help="Path name of pipeline configuration file of " "mxManufacture.",
)

flags.DEFINE_string(
    name="infer_stream_name",
    default=None,
    help="Infer stream name configured in pipeline "
    "configuration file of mxManufacture",
)

flags.DEFINE_boolean(
    name="draw_box",
    default=True,
    help="Whether out put the inferred image with bounding box",
)

flags.DEFINE_enum(
    name="preprocess",
    default="OPENCV",
    enum_values=["DVPP", "OPENCV"],
    help="Preprocess method to use, default OpenCV.",
)

flags.DEFINE_boolean(
    name="coco",
    default=True,
    help="Whether use coco dataset to test performance.",
)

flags.DEFINE_float(
    name="score_thresh_for_draw",
    default=0.5,
    help="Draw bounding box if the confidence greater than.",
)

flags.DEFINE_string(
    name="output_dir",
    default=None,
    help="Where to out put the inferred image with bounding box, if the "
    "draw_box is set, this parameter must be set.",
)

flags.DEFINE_integer(
    name="how_many_images_to_infer",
    default=-1,
    help="Infer how many images in img_dir, -1 means all.",
)

flags.DEFINE_integer(
    name="infer_timeout_secs",
    default=3,
    help="Time out(in seconds) to get the infer result. ",
)

flags.DEFINE_integer(
    name="model_input_height",
    default=640,
    help="Image height input to " "model.",
)

flags.DEFINE_integer(
    name="model_input_width", default=640, help="Image width input to model."
)

flags.DEFINE_integer(
    name="display_step",
    default=100,
    help="Every how many images to print the inference real speed and "
    "progress.",
)

flags.mark_flag_as_required("img_dir")
flags.mark_flag_as_required("pipeline_config")
flags.mark_flag_as_required("infer_stream_name")
flags.mark_flag_as_required("output_dir")


def draw_image(input_image, bboxes, output_img):
    # 原图
    image = cv2.imread(input_image)

    # 模型推理输出数据，需要往后处理代码中增加几行输出文档的代码
    color_index_dict = {
        0: (0, 0, 255),
        1: (0, 255, 0),
        2: (255, 0, 0),
        3: (255, 255, 0),
        4: (255, 0, 255),
        5: (0, 255, 255),
        6: (255, 128, 0),
        7: (128, 128, 255),
        8: (0, 255, 128),
        9: (128, 128, 0),
    }
    for index, bbox in enumerate(bboxes):
        color_key = index % 10
        color = color_index_dict.get(color_key)
        # Coordinate must be integer.
        # bbox = list(map(lambda cor: int(cor), bbox))
        # pdb.set_trace()
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

    # 新的图片
    cv2.imwrite(output_img, image)


def draw_img_fun(img_id, bboxes):
    img_name = "%012d.jpg" % img_id
    input_img_dir = FLAGS.img_dir
    input_img = os.path.join(input_img_dir, img_name)

    boxed_img = os.path.join(BOXED_IMG_DIR, img_name)
    draw_image(input_img, bboxes, boxed_img)


def trans_class_id(k):
    c = None
    if 1 <= k <= 11:
        c = k
    elif 12 <= k <= 24:
        c = k + 1
    elif 25 <= k <= 26:
        c = k + 2
    elif 27 <= k <= 40:
        c = k + 4
    elif 41 <= k <= 60:
        c = k + 5
    elif k == 61:
        c = k + 6
    elif k == 62:
        c = k + 8
    elif 63 <= k <= 73:
        c = k + 9
    elif 74 <= k <= 80:
        c = k + 10
    return c


def parse_result(img_id, json_content):
    obj_list = json.loads(json_content).get("MxpiObject", [])
    pic_infer_dict_list = []
    bboxes_for_drawing = []
    txt_lines_list = []
    hratio, wratio = det_restore_ratio.get(img_id, (1, 1))
    for o in obj_list:
        x0, y0, x1, y1 = (
            round(o.get("x0"), 4),
            round(o.get("y0"), 4),
            round(o.get("x1"), 4),
            round(o.get("y1"), 4),
        )
        # For MAP
        bbox_for_map = [
            int(x0 * wratio),
            int(y0 * hratio),
            int((x1 - x0) * wratio),
            int((y1 - y0) * hratio),
        ]
        # For drawing bounding box.
        bbox_for_drawing = [int(x0), int(y0), int(x1), int(y1)]
        # calculation
        tmp_list = [
            o.get("classVec")[0].get("classId"),
            o.get("classVec")[0].get("confidence"),
            x0,
            y0,
            x1,
            y1,
        ]
        tmp_list = map(str, tmp_list)
        txt_lines_list.append(" ".join(tmp_list))
        category_id = o.get("classVec")[0].get("classId")  # 1-80, GT:1-90
        category_id = trans_class_id(category_id)
        score = o.get("classVec")[0].get("confidence")

        pic_infer_dict_list.append(
            dict(
                image_id=img_id,
                bbox=bbox_for_map,
                category_id=category_id,
                score=score,
            )
        )

        if FLAGS.draw_box and score > FLAGS.score_thresh_for_draw:
            bboxes_for_drawing.append(bbox_for_drawing[:])

    txt_name = "%012d.txt" % img_id
    txt_full_name = os.path.join(TXT_DIR, txt_name)
    with open(txt_full_name, "w") as fw:
        fw.write("\n".join(txt_lines_list))
        fw.write("\n")

    if FLAGS.draw_box:
        draw_img_fun(img_id, bboxes_for_drawing)

    return pic_infer_dict_list


def send_img_with_opencv_handled(stream_manager_api, img_file_name):
    img = cv2.imread(img_file_name)
    height = img.shape[0]
    width = img.shape[1]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (FLAGS.model_input_width, FLAGS.model_input_height))

    img_id = (
        int(os.path.basename(img_file_name).split(".")[0])
        if FLAGS.coco
        else img_file_name
    )
    # """
    # height/FLAGS.model_input_height = hx/ DH =>hx = DH * (
    # height/FLAGS.model_input_height)
    # """
    det_restore_ratio[img_id] = (
        round(height * 1.0 / FLAGS.model_input_height, 4),
        round(width * 1.0 / FLAGS.model_input_width, 4),
    )
    array_bytes = img.tobytes()
    data_input = MxDataInput()
    data_input.data = array_bytes
    key = b"appsrc0"
    protobuf_vec = InProtobufVector()

    vision_list = MxpiDataType.MxpiVisionList()
    vision_vec = vision_list.visionVec.add()
    vision_vec.visionInfo.format = 1
    vision_vec.visionInfo.width = FLAGS.model_input_width
    vision_vec.visionInfo.height = FLAGS.model_input_height
    vision_vec.visionInfo.widthAligned = FLAGS.model_input_width
    vision_vec.visionInfo.heightAligned = FLAGS.model_input_height
    vision_vec.visionData.deviceId = 0
    vision_vec.visionData.memType = 0
    vision_vec.visionData.dataStr = data_input.data

    protobuf = MxProtobufIn()
    protobuf.key = key
    protobuf.type = b"MxTools.MxpiVisionList"
    protobuf.protobuf = vision_list.SerializeToString()

    protobuf_vec.push_back(protobuf)

    unique_id = stream_manager_api.SendProtobuf(
        FLAGS.infer_stream_name.encode("utf8"), 0, protobuf_vec
    )

    if unique_id < 0:
        print("Failed to send data to stream.")
        exit()

    key_vec = StringVector()
    key_vec.push_back(b"mxpi_modelinfer0")
    return unique_id


def display_infer_progress(img_num, index, report_file, start_secs):
    cur_secs = time.time()
    acc_secs = round(cur_secs - start_secs, 4)
    real_speed = round((cur_secs - start_secs) * 1000 / (index + 1), 4)
    perf_detail = (
        f"Inferred: {index + 1}/{img_num} images; "
        f"took: {acc_secs} seconds; "
        f"average inference speed at: {real_speed} ms/image\n"
    )
    print(perf_detail)
    threading.Thread(
        target=write_speed_detail, args=(perf_detail, report_file)
    ).start()


def write_speed_detail(perf_detail, report_file):
    report_file.write(perf_detail)
    report_file.flush()


def handle_infer_result(all_infer_dict_list, img_id, infer_result, img_ext="jpg"):
    if infer_result.errorCode != 0:
        print(
            "GetResultWithUniqueId error. errorCode=%d, errorMsg=%s"
            % (infer_result.errorCode, infer_result.data.decode())
        )
        exit()

    info_json_str = infer_result.data.decode()
    with infer_ret_list_lock:
        all_infer_dict_list.extend(parse_result(img_id, info_json_str))


def infer_imgs_in_dir_with_open_cv():
    input_dir = FLAGS.img_dir
    report_file = open(PERF_REPORT_TXT, "a+")
    imgs = [
        img_name
        for img_name in os.listdir(input_dir)
        if "boxed" not in img_name
        and img_name.lower().endswith((".jpg", ".jpeg"))
    ]

    img_file_names = [
        os.path.join(input_dir, img_name)
        for img_name in imgs
        if "boxed" not in img_name
    ]
    all_infer_dict_list = []
    stream_manager_api = prepare_infer_stream()
    start_secs = time.time()
    img_num = len(img_file_names)
    parse_det_threads = []
    for index, img_file_name in enumerate(img_file_names):
        inferred_cnt = index + 1
        send_img_with_opencv_handled(stream_manager_api, img_file_name)
        infer_result = stream_manager_api.GetResult(
            FLAGS.infer_stream_name.encode("utf8"), 0
        )

        if inferred_cnt % FLAGS.display_step == 0:
            display_infer_progress(img_num, index, report_file, start_secs)

        name, ext = os.path.splitext(os.path.basename(img_file_name))
        img_id = int(name) if FLAGS.coco else name

        t = threading.Thread(
            target=handle_infer_result,
            args=(all_infer_dict_list, img_id, infer_result, ext),
        )
        t.start()
        parse_det_threads.append(t)

        if inferred_cnt >= FLAGS.how_many_images_to_infer > 0:
            img_num = inferred_cnt
            print(f"Inferred all {inferred_cnt} images to SDK success.")
            break

    for t in parse_det_threads:
        t.join()

    finish_secs = time.time()
    avg_infer_speed = round((finish_secs - start_secs) * 1000 / img_num, 4)
    final_perf = (
        f"Infer with OPENCV finished, average speed:{avg_infer_speed} "
        f"ms/image for {img_num} images.\n\n"
    )
    print(final_perf)
    report_file.write(final_perf)
    report_file.close()

    with open(DET_RESULT_JSON, "w") as fw:
        fw.write(json.dumps(all_infer_dict_list))

    stream_manager_api.DestroyAllStreams()


def send_many_images(stream_manager_api):

    input_dir = FLAGS.img_dir

    imgs = os.listdir(input_dir)
    img_ids = map(lambda img_name: int(img_name.split(".")[0]), imgs)
    img_file_names = [
        os.path.join(input_dir, img_name)
        for img_name in imgs
        if "boxed" not in img_name
    ]
    infer_cnt = (
        len(img_file_names)
        if FLAGS.how_many_images_to_infer == -1
        else FLAGS.how_many_images_to_infer
    )
    start = time.time()
    uuid_list = []
    for img_file_name in img_file_names[:infer_cnt]:
        data_input = MxDataInput()
        with open(img_file_name, "rb") as f:
            data_input.data = f.read()

        in_plugin_id = 0
        unique_id = stream_manager_api.SendDataWithUniqueId(
            FLAGS.infer_stream_name.encode("utf8"), in_plugin_id, data_input
        )
        if unique_id < 0:
            print("Failed to send data to stream.")
            exit()

        uuid_list.append(unique_id)

    end = time.time()
    time_str = (
        f"\nSend all images data took: {round((end-start)*1000, 2)} ms\n"
    )
    print(time_str)
    with open(PERF_REPORT_TXT, "a+") as fw:
        fw.write(time_str)

    return zip(uuid_list, img_ids)


def get_all_images_result(uuid_img_id_zip, stream_manager_api):
    start_secs = time.time()
    all_infer_dict_list = []
    report_file = open(PERF_REPORT_TXT, "a+")
    img_num = len(
        [
            img_name
            for img_name in os.listdir(FLAGS.img_dir)
            if "boxed" not in img_name
        ]
    )
    for index, (uuid, img_id) in enumerate(uuid_img_id_zip):
        infer_result = stream_manager_api.GetResultWithUniqueId(
            FLAGS.infer_stream_name.encode("utf8"),
            uuid,
            FLAGS.infer_timeout_secs * 1000,
        )
        if (index + 1) % FLAGS.display_step == 0:
            cur_secs = time.time()
            acc_secs = round(cur_secs - start_secs, 4)
            real_speed = round((cur_secs - start_secs) * 1000 / (index + 1), 4)
            perf_detail = (
                f"Inferred: {index + 1}/{img_num} images; "
                f"took: {acc_secs} seconds; "
                f"average inference speed at: {real_speed} ms/image\n"
            )
            print(perf_detail)
            threading.Thread(
                target=write_speed_detail, args=(perf_detail, report_file)
            ).start()

        threading.Thread(
            target=parse_infer_result,
            args=(all_infer_dict_list, img_id, infer_result),
        ).start()

    finish_secs = time.time()
    avg_infer_speed = round((finish_secs - start_secs) * 1000 / img_num, 4)
    final_perf = (
        f"Infer finished, average speed:{avg_infer_speed} "
        f"ms/image for {img_num} images.\n\n"
    )
    report_file.write(final_perf)
    report_file.close()

    return all_infer_dict_list



def parse_infer_result(all_infer_dict_list, img_id, infer_result):
    if infer_result.errorCode != 0:
        print(
            "GetResultWithUniqueId error. errorCode=%d, errorMsg=%s"
            % (infer_result.errorCode, infer_result.data.decode())
        )
        exit()

    info_json_str = infer_result.data.decode()
    img_infer_ret = parse_result(img_id, info_json_str)
    with infer_ret_list_lock:
        all_infer_dict_list.extend(img_infer_ret)


def infer_img(stream_manager_api, input_image, infer_stream_name):
    """Infer one input image with specified stream name configured in
    mxManufacture pipeline config file.

    :param stream_manager_api:
    :param input_image: file name the image to be inferred.
    :param infer_stream_name:
    :return:
    """
    data_input = MxDataInput()
    with open(input_image, "rb") as f:
        data_input.data = f.read()

    in_plugin_id = 0
    unique_id = stream_manager_api.SendDataWithUniqueId(
        infer_stream_name.encode("utf8"), in_plugin_id, data_input
    )
    if unique_id < 0:
        print("Failed to send data to stream.")
        exit()

    # Obtain the inference result by specifying streamName and unique_id.
    infer_result = stream_manager_api.GetResultWithUniqueId(
        infer_stream_name.encode("utf8"), unique_id, 3000
    )
    end = time.time()
    print(f"Infer time: {end} s.")
    if infer_result.errorCode != 0:
        print(
            "GetResultWithUniqueId error. errorCode=%d, errorMsg=%s"
            % (infer_result.errorCode, infer_result.data.decode())
        )
        exit()

    info_json_str = infer_result.data.decode()
    img_id = int(os.path.basename(input_image).split(".")[0])
    return parse_result(img_id, info_json_str)


def prepare_infer_stream():
    # init stream manager
    stream_manager_api = StreamManagerApi()
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()
    # create streams by pipeline config file
    with open(FLAGS.pipeline_config, "rb") as f:
        pipelineStr = f.read()

    ret = stream_manager_api.CreateMultipleStreams(pipelineStr)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()
    return stream_manager_api


def infer_imgs():
    stream_manager_api = prepare_infer_stream()
    uuid_img_id_zip = send_many_images(stream_manager_api)
    all_infer_dict_list = get_all_images_result(
        uuid_img_id_zip, stream_manager_api
    )
    with open(DET_RESULT_JSON, "w") as fw:
        fw.write(json.dumps(all_infer_dict_list))

    stream_manager_api.DestroyAllStreams()


def main(unused_arg):
    global BOXED_IMG_DIR
    global TXT_DIR
    global PERF_REPORT_TXT
    global DET_RESULT_JSON
    # '''
    # output_dir
    # |_boxed_imgs
    # |_txts
    # |_per_report_npu.txt
    # |_det_result_npu.json
    # '''

    BOXED_IMG_DIR = os.path.join(FLAGS.output_dir, "boxed_imgs")
    TXT_DIR = os.path.join(FLAGS.output_dir, "txts")
    PERF_REPORT_TXT = os.path.join(FLAGS.output_dir, "om_perf_report.txt")
    DET_RESULT_JSON = os.path.join(FLAGS.output_dir, "om_det_result.json")

    if not os.path.exists(FLAGS.output_dir):
        os.makedirs(FLAGS.output_dir)

    if not os.path.exists(TXT_DIR):
        os.makedirs(TXT_DIR)

    if FLAGS.draw_box and not os.path.exists(BOXED_IMG_DIR):
        os.makedirs(BOXED_IMG_DIR)

    now_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    head_info = f"{'-'*50}Perf Test On NPU starts @ {now_time_str}{'-'*50}\n"
    with open(PERF_REPORT_TXT, "a+") as fw:
        fw.write(head_info)

    if FLAGS.preprocess == "DVPP":
        print("Start DVPP infer pert testing...")
        infer_imgs()
    else:
        print("Start OpenCV infer pert testing...")
        infer_imgs_in_dir_with_open_cv()

    end_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    tail_info = f"{'-'*50}Perf Test On NPU ends @ {end_time_str}{'-'*50}\n"
    with open(PERF_REPORT_TXT, "a+") as fw:
        fw.write(tail_info)


if __name__ == "__main__":
    app.run(main)
