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
import time
import datetime
import random
import sys
import requests
import flatbuffers
import numpy as np
from mindspore.schema import (RequestFLJob, ResponseFLJob, ResponseCode,
                              RequestUpdateModel, FeatureMap, RequestGetModel, ResponseGetModel)

parser = argparse.ArgumentParser()
parser.add_argument("--pid", type=int, default=0)
parser.add_argument("--http_ip", type=str, default="10.113.216.106")
parser.add_argument("--http_port", type=int, default=6666)
parser.add_argument("--use_elb", type=bool, default=False)
parser.add_argument("--server_num", type=int, default=1)

args, _ = parser.parse_known_args()
pid = args.pid
http_ip = args.http_ip
http_port = args.http_port
use_elb = args.use_elb
server_num = args.server_num

str_fl_id = 'fl_lenet_' + str(pid)

def generate_port():
    if not use_elb:
        return http_port
    port = random.randint(0, 100000) % server_num + http_port
    return port


def build_start_fl_job(iteration):
    start_fl_job_builder = flatbuffers.Builder(1024)

    fl_name = start_fl_job_builder.CreateString('fl_test_job')
    fl_id = start_fl_job_builder.CreateString(str_fl_id)
    data_size = 32
    timestamp = start_fl_job_builder.CreateString('2020/11/16/19/18')

    RequestFLJob.RequestFLJobStart(start_fl_job_builder)
    RequestFLJob.RequestFLJobAddFlName(start_fl_job_builder, fl_name)
    RequestFLJob.RequestFLJobAddFlId(start_fl_job_builder, fl_id)
    RequestFLJob.RequestFLJobAddIteration(start_fl_job_builder, iteration)
    RequestFLJob.RequestFLJobAddDataSize(start_fl_job_builder, data_size)
    RequestFLJob.RequestFLJobAddTimestamp(start_fl_job_builder, timestamp)
    fl_job_req = RequestFLJob.RequestFLJobEnd(start_fl_job_builder)

    start_fl_job_builder.Finish(fl_job_req)
    buf = start_fl_job_builder.Output()
    return buf

def build_feature_map(builder, names, lengths):
    if len(names) != len(lengths):
        return None
    feature_maps = []
    np_data = []
    for j, _ in enumerate(names):
        name = names[j]
        length = lengths[j]
        weight_full_name = builder.CreateString(name)
        FeatureMap.FeatureMapStartDataVector(builder, length)
        weight = np.random.rand(length) * 32
        np_data.append(weight)
        for idx in range(length - 1, -1, -1):
            builder.PrependFloat32(weight[idx])
        data = builder.EndVector(length)
        FeatureMap.FeatureMapStart(builder)
        FeatureMap.FeatureMapAddData(builder, data)
        FeatureMap.FeatureMapAddWeightFullname(builder, weight_full_name)
        feature_map = FeatureMap.FeatureMapEnd(builder)
        feature_maps.append(feature_map)
    return feature_maps, np_data

def build_update_model(iteration):
    builder_update_model = flatbuffers.Builder(1)
    fl_name = builder_update_model.CreateString('fl_test_job')
    fl_id = builder_update_model.CreateString(str_fl_id)
    timestamp = builder_update_model.CreateString('2020/11/16/19/18')

    feature_maps, np_data = build_feature_map(builder_update_model,
                                              ["conv1.weight", "conv2.weight", "fc1.weight",
                                               "fc2.weight", "fc3.weight", "fc1.bias", "fc2.bias", "fc3.bias"],
                                              [450, 2400, 48000, 10080, 5208, 120, 84, 62])

    RequestUpdateModel.RequestUpdateModelStartFeatureMapVector(builder_update_model, 1)
    for single_feature_map in feature_maps:
        builder_update_model.PrependUOffsetTRelative(single_feature_map)
    feature_map = builder_update_model.EndVector(len(feature_maps))

    RequestUpdateModel.RequestUpdateModelStart(builder_update_model)
    RequestUpdateModel.RequestUpdateModelAddFlName(builder_update_model, fl_name)
    RequestUpdateModel.RequestUpdateModelAddFlId(builder_update_model, fl_id)
    RequestUpdateModel.RequestUpdateModelAddIteration(builder_update_model, iteration)
    RequestUpdateModel.RequestUpdateModelAddFeatureMap(builder_update_model, feature_map)
    RequestUpdateModel.RequestUpdateModelAddTimestamp(builder_update_model, timestamp)
    req_update_model = RequestUpdateModel.RequestUpdateModelEnd(builder_update_model)
    builder_update_model.Finish(req_update_model)
    buf = builder_update_model.Output()
    return buf, np_data

def build_get_model(iteration):
    builder_get_model = flatbuffers.Builder(1)
    fl_name = builder_get_model.CreateString('fl_test_job')
    timestamp = builder_get_model.CreateString('2020/12/16/19/18')

    RequestGetModel.RequestGetModelStart(builder_get_model)
    RequestGetModel.RequestGetModelAddFlName(builder_get_model, fl_name)
    RequestGetModel.RequestGetModelAddIteration(builder_get_model, iteration)
    RequestGetModel.RequestGetModelAddTimestamp(builder_get_model, timestamp)
    req_get_model = RequestGetModel.RequestGetModelEnd(builder_get_model)
    builder_get_model.Finish(req_get_model)
    buf = builder_get_model.Output()
    return buf

def datetime_to_timestamp(datetime_obj):
    """将本地(local) datetime 格式的时间 (含毫秒) 转为毫秒时间戳
    :param datetime_obj: {datetime}2016-02-25 20:21:04.242000
    :return: 13 位的毫秒时间戳  1456402864242
    """
    local_timestamp = time.mktime(datetime_obj.timetuple()) * 1000.0 + datetime_obj.microsecond // 1000.0
    return local_timestamp

weight_to_idx = {
    "conv1.weight": 0,
    "conv2.weight": 1,
    "fc1.weight": 2,
    "fc2.weight": 3,
    "fc3.weight": 4,
    "fc1.bias": 5,
    "fc2.bias": 6,
    "fc3.bias": 7
}

session = requests.Session()
current_iteration = 1
url = "http://" + http_ip + ":" + str(generate_port())
np.random.seed(0)
while True:
    url1 = "http://" + http_ip + ":" + str(generate_port()) + '/startFLJob'
    print("start url is ", url1)
    x = session.post(url1, data=build_start_fl_job(current_iteration))
    while x.text == "The cluster is in safemode.":
        x = session.post(url1, data=build_start_fl_job(current_iteration))

    rsp_fl_job = ResponseFLJob.ResponseFLJob.GetRootAsResponseFLJob(x.content, 0)
    while rsp_fl_job.Retcode() != ResponseCode.ResponseCode.SUCCEED:
        x = session.post(url1, data=build_start_fl_job(current_iteration))
        while x.text == "The cluster is in safemode.":
            time.sleep(0.2)
            x = session.post(url1, data=build_start_fl_job(current_iteration))
        rsp_fl_job = ResponseFLJob.ResponseFLJob.GetRootAsResponseFLJob(x.content, 0)
    print("epoch is", rsp_fl_job.FlPlanConfig().Epochs())
    print("iteration is", rsp_fl_job.Iteration())
    current_iteration = rsp_fl_job.Iteration()
    sys.stdout.flush()

    url2 = "http://" + http_ip + ":" + str(generate_port()) + '/updateModel'
    print("req update model iteration:", current_iteration, ", id:", args.pid)
    update_model_buf, update_model_np_data = build_update_model(current_iteration)
    x = session.post(url2, data=update_model_buf)
    while x.text == "The cluster is in safemode.":
        time.sleep(0.2)
        x = session.post(url1, data=update_model_buf)

    print("rsp update model iteration:", current_iteration, ", id:", args.pid)
    sys.stdout.flush()

    url3 = "http://" + http_ip + ":" + str(generate_port()) + '/getModel'
    print("req get model iteration:", current_iteration, ", id:", args.pid)
    x = session.post(url3, data=build_get_model(current_iteration))
    while x.text == "The cluster is in safemode.":
        time.sleep(0.2)
        x = session.post(url3, data=build_get_model(current_iteration))

    rsp_get_model = ResponseGetModel.ResponseGetModel.GetRootAsResponseGetModel(x.content, 0)
    print("rsp get model iteration:", current_iteration, ", id:", args.pid, rsp_get_model.Retcode())
    sys.stdout.flush()

    next_req_timestamp = 0
    if rsp_get_model.Retcode() == ResponseCode.ResponseCode.OutOfTime:
        next_req_timestamp = int(rsp_get_model.Timestamp().decode('utf-8'))
        print("Last iteration is invalid, next request timestamp:", next_req_timestamp)
        sys.stdout.flush()
    elif rsp_get_model.Retcode() == ResponseCode.ResponseCode.SucNotReady:
        repeat_time = 0
        while rsp_get_model.Retcode() == ResponseCode.ResponseCode.SucNotReady:
            time.sleep(0.2)
            x = session.post(url3, data=build_get_model(current_iteration))
            while x.text == "The cluster is in safemode.":
                time.sleep(0.2)
                x = session.post(url3, data=build_get_model(current_iteration))

            rsp_get_model = ResponseGetModel.ResponseGetModel.GetRootAsResponseGetModel(x.content, 0)
            if rsp_get_model.Retcode() == ResponseCode.ResponseCode.OutOfTime:
                next_req_timestamp = int(rsp_get_model.Timestamp().decode('utf-8'))
                print("Last iteration is invalid, next request timestamp:", next_req_timestamp)
                sys.stdout.flush()
                break
            repeat_time += 1
            if repeat_time > 1000:
                print("GetModel try timeout ", args.pid)
                sys.exit(0)
    else:
        pass

    if next_req_timestamp == 0:
        for i in range(0, 1):
            print(rsp_get_model.FeatureMap(i).WeightFullname())
            origin = update_model_np_data[weight_to_idx[rsp_get_model.FeatureMap(i).WeightFullname().decode('utf-8')]]
            after = rsp_get_model.FeatureMap(i).DataAsNumpy() * 32
            print("Before update model", args.pid, origin[0:10])
            print("After get model", args.pid, after[0:10])
            sys.stdout.flush()
            assert np.allclose(origin, after, rtol=1e-05, atol=1e-05)
    else:
        # Sleep to the next request timestamp
        current_ts = datetime_to_timestamp(datetime.datetime.now())
        duration = next_req_timestamp - current_ts
        if duration > 0:
            time.sleep(duration / 1000)
