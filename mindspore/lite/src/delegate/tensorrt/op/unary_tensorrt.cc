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

#include "src/delegate/tensorrt/op/unary_tensorrt.h"

namespace mindspore::lite {
int UnaryTensorRT::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                             const std::vector<mindspore::MSTensor> &out_tensors) {
  if (!IsShapeKnown()) {
    MS_LOG(ERROR) << "Unsupported input tensor unknown shape: " << op_name_;
    return RET_ERROR;
  }
  if (in_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
  }
  auto it = unary_ops_.find(primitive->value_type());
  if (it != unary_ops_.end()) {
    unary_op_ = it->second;
  } else {
    MS_LOG(ERROR) << "unsupported unary ops type: " << schema::EnumNamePrimitiveType(primitive->value_type());
    return RET_ERROR;
  }
  return RET_OK;
}

int UnaryTensorRT::AddInnerOp(nvinfer1::INetworkDefinition *network) {
  if (network == nullptr || this->tensorrt_in_tensors_.size() != 1) {
    MS_LOG(ERROR) << "network or input tensor is invalid";
    return RET_ERROR;
  }
  nvinfer1::IUnaryLayer *cal_layer = network->addUnary(*tensorrt_in_tensors_[0].trt_tensor_, unary_op_);
  if (cal_layer == nullptr) {
    MS_LOG(ERROR) << "addUnary failed for: " << op_name_;
    return RET_ERROR;
  }
  cal_layer->setName(op_name_.c_str());

  nvinfer1::ITensor *op_out_tensor = cal_layer->getOutput(0);
  op_out_tensor->setName((op_name_ + "_output").c_str());
  this->AddInnerOutTensors(ITensorHelper{op_out_tensor, tensorrt_in_tensors_[0].format_});
  return RET_OK;
}
}  // namespace mindspore::lite
