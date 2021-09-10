# Copyright 2020-2021 Huawei Technologies Co., Ltd
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
from StreamManagerApi import StreamManagerApi, MxDataInput


def read_file_list(input_file):
    """
    :param infer file content:
        0 xxx/xxx/a.jpg 1920 1080 0 453 369 473 391 1 588 245 608 268
        1 xxx/xxx/b.jpg 1920 1080 1 466 403 485 422 2 793 300 809 320
        ...
    :return image path list
    """
    image_file_list = []
    if not os.path.exists(input_file):
        print('input file does not exists.')
    with open(input_file, "r") as fs:
        for line in fs.readlines():
            line = line.strip('\n').split(' ')[1]
            image_file_list.append(line)
    return image_file_list


def save_infer_result(result_dir, result):
    """
    save infer result to the file, Write format:
        Object detected num is 5
        #Obj: 1, box: 453 369 473 391, confidence: 0.3, label: person, id: 0
        ...
    :param result_dir is the dir of save result
    :param result content bbox and class_id of all object
    """
    load_dict = json.loads(result)
    if load_dict.get('MxpiObject') is None:
        with open(result_dir + '/result.txt', 'a+') as f_write:
            f_write.write("")
    else:
        res_vec = load_dict.get('MxpiObject')
        with open(result_dir + '/result.txt', 'a+') as f_write:
            object_list = 'Object detected num is ' + str(len(res_vec)) + '\n'
            f_write.writelines(object_list)
            for index, object_item in enumerate(res_vec):
                class_info = object_item.get('classVec')[0]
                object_info = '#Obj: ' + str(index) + ', box: ' + \
                              str(object_item.get('x0')) + ' ' + \
                              str(object_item.get('y0')) + ' ' + \
                              str(object_item.get('x1')) + ' ' + \
                              str(object_item.get('y1')) + ', confidence: ' + \
                              str(class_info.get('confidence')) + ', label: ' + \
                              class_info.get('className') + ', id: ' + \
                              str(class_info.get('classId')) + '\n'
                f_write.writelines(object_info)


if __name__ == '__main__':
    # init stream manager
    stream_manager = StreamManagerApi()
    ret = stream_manager.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    with open("./config/yolov4_tiny.pipeline", 'rb') as f:
        pipeline = f.read()
    ret = stream_manager.CreateMultipleStreams(pipeline)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # Construct the input of the stream
    data_input = MxDataInput()

    infer_file = './coco2017_minival.txt'
    file_list = read_file_list(infer_file)
    res_dir_name = 'result'
    if not os.path.exists(res_dir_name):
        os.makedirs(res_dir_name)

    for file_path in file_list:
        print(file_path)
        file_name = file_path.split('/')[-1]
        if not (file_name.lower().endswith(".jpg") or file_name.lower().endswith(".jpeg")):
            continue

        with open(file_path, 'rb') as f:
            data_input.data = f.read()

        # Inputs data to a specified stream based on streamName.
        stream_name = b'im_yolov4tiny'
        inplugin_id = 0
        unique_id = stream_manager.SendData(stream_name, inplugin_id, data_input)
        if unique_id < 0:
            print("Failed to send data to stream.")
            exit()
        # Obtain the inference result by specifying streamName and uniqueId.
        infer_result = stream_manager.GetResult(stream_name, unique_id)
        if infer_result.errorCode != 0:
            print("GetResultWithUniqueId error. errorCode=%d, errorMsg=%s" % (
                infer_result.errorCode, infer_result.data.decode()))
            exit()
        save_infer_result(res_dir_name, infer_result.data.decode())


    # destroy streams
    stream_manager.DestroyAllStreams()
