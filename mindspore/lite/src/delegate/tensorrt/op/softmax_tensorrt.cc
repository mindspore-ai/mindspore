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
  if (!IsShapeKnown()) {
    MS_LOG(ERROR) << "Unsupported input tensor unknown shape: " << op_name_;
    return RET_ERROR;
  }
  softmax_op_ = primitive->value_as_Softmax();
  if (softmax_op_ == nullptr) {
    MS_LOG(ERROR) << "convert failed";
    return RET_ERROR;
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
  out_tensor->setName((op_name_ + "_output").c_str());
  this->AddInnerOutTensors(ITensorHelper{out_tensor, tensorrt_in_tensors_[0].format_});
  return RET_OK;
}

nvinfer1::ISoftMaxLayer *SoftMaxTensorRT::AddSoftMaxOp(nvinfer1::INetworkDefinition *network) {
  nvinfer1::ISoftMaxLayer *current_layer_ = network->addSoftMax(*tensorrt_in_tensors_[0].trt_tensor_);
  if (current_layer_ == nullptr) {
    MS_LOG(ERROR) << "add softmax op failed for TensorRT.";
    return nullptr;
  }
  auto axis = softmax_op_->axis();
  auto axis_val = std::vector<int64_t>(axis->begin(), axis->end());

  if (axis_val.size() != 1) {
    MS_LOG(WARNING) << "axis needs check";
  }

  if (axis_val[0] >= this->tensorrt_in_tensors_[0].trt_tensor_->getDimensions().nbDims) {
    MS_LOG(ERROR) << "axis is larger than input tensor dims.";
    return nullptr;
  }
  int64_t axis_format_value = axis_val[0];
  if (tensorrt_in_tensors_[0].trt_tensor_->getDimensions().nbDims == DIMENSION_4D &&
      tensorrt_in_tensors_[0].format_ == Format::NCHW) {
    // transpose axis to NCHW
    axis_format_value = ConvertAxisFromNHWC2NCHW(axis_val[0]);
  }
  uint32_t axis_bit = 1 << axis_format_value;
  MS_LOG(DEBUG) << op_name_ << " set axis to " << axis_bit;
  current_layer_->setAxes(axis_bit);
  return current_layer_;
}
}  // namespace mindspore::lite
