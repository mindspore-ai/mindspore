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
import sys
import grpc
import numpy as np
import ms_service_pb2
import ms_service_pb2_grpc


def run():
    if len(sys.argv) > 2:
        sys.exit("input error")
    channel_str = ""
    if len(sys.argv) == 2:
        split_args = sys.argv[1].split('=')
        if len(split_args) > 1:
            channel_str = split_args[1]
        else:
            channel_str = 'localhost:5500'
    else:
        channel_str = 'localhost:5500'

    channel = grpc.insecure_channel(channel_str)
    stub = ms_service_pb2_grpc.MSServiceStub(channel)
    request = ms_service_pb2.PredictRequest()

    x = request.data.add()
    x.tensor_shape.dims.extend([4])
    x.tensor_type = ms_service_pb2.MS_FLOAT32
    x.data = (np.ones([4]).astype(np.float32)).tobytes()

    y = request.data.add()
    y.tensor_shape.dims.extend([4])
    y.tensor_type = ms_service_pb2.MS_FLOAT32
    y.data = (np.ones([4]).astype(np.float32)).tobytes()

    try:
        result = stub.Predict(request)
        result_np = np.frombuffer(result.result[0].data, dtype=np.float32).reshape(result.result[0].tensor_shape.dims)
        print("ms client received: ")
        print(result_np)
    except grpc.RpcError as e:
        print(e.details())
        status_code = e.code()
        print(status_code.name)
        print(status_code.value)

if __name__ == '__main__':
    run()
