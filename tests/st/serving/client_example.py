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

import random
import json
import requests
import grpc
import numpy as np
import ms_service_pb2
import ms_service_pb2_grpc
import mindspore.dataset as de
from mindspore import Tensor, context
from mindspore import log as logger
from tests.st.networks.models.bert.src.bert_model import BertModel
from .generate_model import AddNet, bert_net_cfg

context.set_context(mode=context.GRAPH_MODE, device_target="Ascend")

random.seed(1)
np.random.seed(1)
de.config.set_seed(1)

def test_add():
    channel = grpc.insecure_channel('localhost:5500')
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

    result = stub.Predict(request)
    result_np = np.frombuffer(result.result[0].data, dtype=np.float32).reshape(result.result[0].tensor_shape.dims)
    print("ms client received: ")
    print(result_np)

    net = AddNet()
    net_out = net(Tensor(np.ones([4]).astype(np.float32)), Tensor(np.ones([4]).astype(np.float32)))
    print("add net out: ")
    print(net_out)
    assert np.allclose(net_out.asnumpy(), result_np, 0.001, 0.001, equal_nan=True)

def test_bert():
    MAX_MESSAGE_LENGTH = 0x7fffffff
    input_ids = np.random.randint(0, 1000, size=(2, 32), dtype=np.int32)
    segment_ids = np.zeros((2, 32), dtype=np.int32)
    input_mask = np.zeros((2, 32), dtype=np.int32)

    # grpc visit
    channel = grpc.insecure_channel('localhost:5500', options=[('grpc.max_send_message_length', MAX_MESSAGE_LENGTH),
                                                               ('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)])
    stub = ms_service_pb2_grpc.MSServiceStub(channel)
    request = ms_service_pb2.PredictRequest()

    x = request.data.add()
    x.tensor_shape.dims.extend([2, 32])
    x.tensor_type = ms_service_pb2.MS_INT32
    x.data = input_ids.tobytes()

    y = request.data.add()
    y.tensor_shape.dims.extend([2, 32])
    y.tensor_type = ms_service_pb2.MS_INT32
    y.data = segment_ids.tobytes()

    z = request.data.add()
    z.tensor_shape.dims.extend([2, 32])
    z.tensor_type = ms_service_pb2.MS_INT32
    z.data = input_mask.tobytes()

    result = stub.Predict(request)
    grpc_result = np.frombuffer(result.result[0].data, dtype=np.float32).reshape(result.result[0].tensor_shape.dims)
    print("ms grpc client received: ")
    print(grpc_result)

    # ms result
    net = BertModel(bert_net_cfg, False)
    bert_out = net(Tensor(input_ids), Tensor(segment_ids), Tensor(input_mask))
    print("bert out: ")
    print(bert_out[0])
    bert_out_size = len(bert_out)

    # compare grpc result
    for i in range(bert_out_size):
        grpc_result = np.frombuffer(result.result[i].data, dtype=np.float32).reshape(result.result[i].tensor_shape.dims)
        logger.info("i:{}, grpc_result:{}, bert_out:{}".
                    format(i, result.result[i].tensor_shape.dims, bert_out[i].asnumpy().shape))
        assert np.allclose(bert_out[i].asnumpy(), grpc_result, 0.001, 0.001, equal_nan=True)

    # http visit
    data = {"tensor": [input_ids.tolist(), segment_ids.tolist(), input_mask.tolist()]}
    url = "http://127.0.0.1:5501"
    input_json = json.dumps(data)
    headers = {'Content-type': 'application/json'}
    response = requests.post(url, data=input_json, headers=headers)
    result = response.text
    result = result.replace('\r', '\\r').replace('\n', '\\n')
    result_json = json.loads(result, strict=False)
    http_result = np.array(result_json['tensor'])
    print("ms http client received: ")
    print(http_result[0][:200])

    # compare http result
    for i in range(bert_out_size):
        logger.info("i:{}, http_result:{}, bert_out:{}".
                    format(i, np.shape(http_result[i]), bert_out[i].asnumpy().shape))
        assert np.allclose(bert_out[i].asnumpy(), http_result[i], 0.001, 0.001, equal_nan=True)
