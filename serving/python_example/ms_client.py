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
import grpc
import numpy as np
import ms_service_pb2
import ms_service_pb2_grpc


def run():
    channel = grpc.insecure_channel('localhost:50051')
    stub = ms_service_pb2_grpc.MSServiceStub(channel)
    # request = ms_service_pb2.PredictRequest()
    # request.name = 'haha'
    # response = stub.Eval(request)
    # print("ms client received: " + response.message)

    request = ms_service_pb2.PredictRequest()
    request.data.tensor_shape.dims.extend([32, 1, 32, 32])
    request.data.tensor_type = ms_service_pb2.MS_FLOAT32
    request.data.data = (np.ones([32, 1, 32, 32]).astype(np.float32) * 0.01).tobytes()

    request.label.tensor_shape.dims.extend([32])
    request.label.tensor_type = ms_service_pb2.MS_INT32
    request.label.data = np.ones([32]).astype(np.int32).tobytes()

    result = stub.Predict(request)
    #result_np = np.frombuffer(result.result.data, dtype=np.float32).reshape(result.result.tensor_shape.dims)
    print("ms client received: ")
    #print(result_np)

    # future_list = []
    # times = 1000
    # for i in range(times):
    #     async_future = stub.Eval.future(request)
    #     future_list.append(async_future)
    #     print("async call, future list add item " + str(i));
    #
    # for i in range(len(future_list)):
    #     async_result = future_list[i].result()
    #     print("ms client async get result of item " + str(i))



if __name__ == '__main__':
    run()
