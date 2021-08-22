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

#include "src/delegate/tensorrt/op/softmax_tensorrt.h"

namespace mindspore::lite {
int SoftMaxTensorRT::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                               const std::vector<mindspore::MSTensor> &out_tensors) {
  if (primitive->value_type() == schema::PrimitiveType::PrimitiveType_LogSoftmax) {
    with_log_ = true;
    auto softmax_op = primitive->value_as_LogSoftmax();
    if (softmax_op == nullptr) {
      MS_LOG(ERROR) << "LogSoftmax convert failed";
      return RET_ERROR;
    }
  } else {
    auto softmax_op = primitive->value_as_Softmax();
    if (softmax_op == nullptr) {
      MS_LOG(ERROR) << "convert failed";
      return RET_ERROR;
    }
  }

  if (in_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
    return RET_ERROR;
  }
  return RET_OK;
}

int SoftMaxTensorRT::AddInnerOp(nvinfer1::INetworkDefinition *network) {
  if (network == nullptr) {
    MS_LOG(ERROR) << "network is invalid";
    return RET_ERROR;
  }
  nvinfer1::ISoftMaxLayer *softmax_layer_ = AddSoftMaxOp(network);
  if (softmax_layer_ == nullptr) {
    MS_LOG(ERROR) << "add softmax op failed for TensorRT.";
    return RET_ERROR;
  }
  softmax_layer_->setName((op_name_ + "_softmax").c_str());

  nvinfer1::ITensor *out_tensor = softmax_layer_->getOutput(0);
  if (out_tensor == nullptr) {
    MS_LOG(ERROR) << "softmax output tensor create failed for TensorRT.";
    return RET_ERROR;
  }
  if (with_log_) {
    nvinfer1::IUnaryLayer *log_layer = network->addUnary(*out_tensor, nvinfer1::UnaryOperation::kLOG);
    if (log_layer == nullptr) {
      MS_LOG(ERROR) << "add log op failed for TensorRT.";
      return RET_ERROR;
    }
    log_layer->setName((op_name_ + "_log").c_str());
    out_tensor = log_layer->getOutput(0);
    if (out_tensor == nullptr) {
      MS_LOG(ERROR) << "softmax log output tensor create failed for TensorRT.";
      return RET_ERROR;
    }
  }
  out_tensor->setName(out_tensors_[0].Name().c_str());
  this->AddInnerOutTensors(out_tensor);
  return RET_OK;
}

nvinfer1::ISoftMaxLayer *SoftMaxTensorRT::AddSoftMaxOp(nvinfer1::INetworkDefinition *network) {
  nvinfer1::ISoftMaxLayer *current_layer_ = network->addSoftMax(*this->GetInnerInTensors()[0]);
  if (current_layer_ == nullptr) {
    MS_LOG(ERROR) << "add softmax op failed for TensorRT.";
    return nullptr;
  }
  std::vector<int64_t> axis_val;
  if (with_log_) {
    auto softmax_op = this->GetPrimitive()->value_as_LogSoftmax();
    if (softmax_op == nullptr) {
      MS_LOG(ERROR) << "LogSoftmax convert failed";
      return nullptr;
    }
    int64_t axis = softmax_op->axis();
    axis_val.push_back(axis);
  } else {
    auto softmax_op = this->GetPrimitive()->value_as_Softmax();
    if (softmax_op == nullptr) {
      MS_LOG(ERROR) << "Softmax convert failed";
      return nullptr;
    }
    auto axis = softmax_op->axis();
    axis_val = std::vector<int64_t>(axis->begin(), axis->end());
  }

  if (axis_val.size() != 1) {
    MS_LOG(WARNING) << "axis needs check";
  }

  if (axis_val[0] >= this->tensorrt_in_tensors_[0]->getDimensions().nbDims) {
    MS_LOG(ERROR) << "axis is larger than input tensor dims.";
    return nullptr;
  }
  current_layer_->setAxes(axis_val[0]);
  return current_layer_;
}
}  // namespace mindspore::lite
