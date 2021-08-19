# coding=utf-8
#
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

import MxpiDataType_pb2 as mxpi_data
from StreamManagerApi import InProtobufVector
from StreamManagerApi import MxProtobufIn
from StreamManagerApi import StreamManagerApi


class SDKInferWrapper:
    def __init__(self):
        self._stream_name = None
        self._stream_mgr_api = StreamManagerApi()

        if self._stream_mgr_api.InitManager() != 0:
            raise RuntimeError("Failed to init stream manager.")

    def load_pipeline(self, pipeline_path):
        with open(pipeline_path, 'r') as f:
            pipeline = json.load(f)

        self._stream_name = list(pipeline.keys())[0].encode()
        if self._stream_mgr_api.CreateMultipleStreams(
                json.dumps(pipeline).encode()) != 0:
            raise RuntimeError("Failed to create stream.")

    def do_infer(self, image):
        tensor_pkg_list = mxpi_data.MxpiTensorPackageList()
        tensor_pkg = tensor_pkg_list.tensorPackageVec.add()
        tensor_vec = tensor_pkg.tensorVec.add()
        tensor_vec.deviceId = 0
        tensor_vec.memType = 0

        for dim in [1, *image.shape]:
            tensor_vec.tensorShape.append(dim)

        input_data = image.tobytes()
        tensor_vec.dataStr = input_data
        tensor_vec.tensorDataSize = len(input_data)

        protobuf_vec = InProtobufVector()
        protobuf = MxProtobufIn()
        protobuf.key = b'appsrc0'
        protobuf.type = b'MxTools.MxpiTensorPackageList'
        protobuf.protobuf = tensor_pkg_list.SerializeToString()
        protobuf_vec.push_back(protobuf)

        unique_id = self._stream_mgr_api.SendProtobuf(
            self._stream_name, 0, protobuf_vec)

        if unique_id < 0:
            raise RuntimeError("Failed to send data to stream.")

        infer_result = self._stream_mgr_api.GetResult(
            self._stream_name, unique_id)

        if infer_result.errorCode != 0:
            raise RuntimeError(
                f"GetResult error. errorCode={infer_result.errorCode}, "
                f"errorMsg={infer_result.data.decode()}")
        return infer_result
