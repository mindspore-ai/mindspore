/**
 * Copyright 2021 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/delegate/tensorrt/op/concate_tensorrt.h"
#include <algorithm>

namespace mindspore::lite {
int ConcateTensorRT::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                               const std::vector<mindspore::MSTensor> &out_tensors) {
  if (in_tensors.size() < 1) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
    return RET_ERROR;
  }
  return RET_OK;
}
int ConcateTensorRT::AddInnerOp(nvinfer1::INetworkDefinition *network) {
  if (network == nullptr) {
    MS_LOG(ERROR) << "network is invalid";
    return RET_ERROR;
  }
  // Concat
  auto concate_op = this->op_primitive_->value_as_Concat();
  if (concate_op == nullptr) {
    MS_LOG(ERROR) << "concate_op convert failed";
    return RET_ERROR;
  }
  MS_LOG(INFO) << "in tensort size of concate: " << tensorrt_in_tensors_.size();
  if (tensorrt_in_tensors_.size() != in_tensors_.size()) {
    MS_LOG(ERROR) << "concate_op in tensor is invalid";
    return RET_ERROR;
  }

  int axis = RET_INVALID_OP_ATTR;
  axis = concate_op->axis();

  nvinfer1::ITensor *trt_input_tensors[tensorrt_in_tensors_.size()];
  std::copy(tensorrt_in_tensors_.begin(), tensorrt_in_tensors_.end(), trt_input_tensors);

  nvinfer1::IConcatenationLayer *concate_layer =
    network->addConcatenation(trt_input_tensors, static_cast<int>(tensorrt_in_tensors_.size()));
  if (concate_layer == nullptr) {
    MS_LOG(ERROR) << "addConcatenation failed for TensorRT.";
    return RET_ERROR;
  }

  if (axis != RET_INVALID_OP_ATTR) {
    concate_layer->setAxis(axis);
  }
  concate_layer->setName(op_name_.c_str());
  this->AddInnerOutTensors(concate_layer->getOutput(0));

  return RET_OK;
}
}  // namespace mindspore::lite
