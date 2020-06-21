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
from concurrent import futures
import time
import grpc
import numpy as np
import ms_service_pb2
import ms_service_pb2_grpc
import test_cpu_lenet
from mindspore import Tensor

class MSService(ms_service_pb2_grpc.MSServiceServicer):
    def Predict(self, request, context):
        request_data = request.data
        request_label = request.label

        data_from_buffer = np.frombuffer(request_data.data, dtype=np.float32)
        data_from_buffer = data_from_buffer.reshape(request_data.tensor_shape.dims)
        data = Tensor(data_from_buffer)

        label_from_buffer = np.frombuffer(request_label.data, dtype=np.int32)
        label_from_buffer = label_from_buffer.reshape(request_label.tensor_shape.dims)
        label = Tensor(label_from_buffer)

        result = test_cpu_lenet.test_lenet(data, label)
        result_reply = ms_service_pb2.PredictReply()
        result_reply.result.tensor_shape.dims.extend(result.shape())
        result_reply.result.data = result.asnumpy().tobytes()
        return result_reply

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=1))
    ms_service_pb2_grpc.add_MSServiceServicer_to_server(MSService(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    try:
        while True:
            time.sleep(60*60*24) # one day in seconds
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()
