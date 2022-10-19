/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "src/extendrt/delegate/tensorrt/op/softmax_tensorrt.h"
#include "ops/softmax.h"

namespace mindspore::lite {
int SoftMaxTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                               const std::vector<TensorInfo> &out_tensors) {
  auto softmax_op = AsOps<ops::Softmax>();
  if (softmax_op != nullptr) {
    auto axis = softmax_op->get_axis();
    axis_val_ = std::vector<int64_t>(axis.begin(), axis.end());
  }

  auto logsoftmax_op = AsOps<ops::LogSoftmax>();
  if (logsoftmax_op != nullptr) {
    auto axis = logsoftmax_op->get_axis();
    axis_val_ = std::vector<int64_t>(1, axis);
  }
  if (axis_val_.size() != 1) {
    MS_LOG(ERROR) << "axis needs check";
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
int SoftMaxTensorRT::AddInnerOp(TensorRTContext *ctx) {
  if (ctx == nullptr || ctx->network() == nullptr) {
    MS_LOG(ERROR) << "network is invalid";
    return RET_ERROR;
  }
  nvinfer1::ISoftMaxLayer *softmax_layer_ = AddSoftMaxOp(ctx);
  if (softmax_layer_ == nullptr) {
    MS_LOG(ERROR) << "add softmax op failed for TensorRT.";
    return RET_ERROR;
  }
  softmax_layer_->setName((op_name_ + "_softmax").c_str());
  this->layer_ = softmax_layer_;

  nvinfer1::ITensor *out_tensor = softmax_layer_->getOutput(0);
  if (out_tensor == nullptr) {
    MS_LOG(ERROR) << "softmax output tensor create failed for TensorRT.";
    return RET_ERROR;
  }
  auto logsoftmax_op = AsOps<ops::LogSoftmax>();
  if (logsoftmax_op != nullptr) {
    out_tensor = ctx->network()->addUnary(*out_tensor, nvinfer1::UnaryOperation::kLOG)->getOutput(0);
  }
  ctx->RegisterTensor(ITensorHelper{out_tensor, input(ctx, 0).format_, input(ctx, 0).same_format_},
                      out_tensors_[0].Name());
  return RET_OK;
}

nvinfer1::ISoftMaxLayer *SoftMaxTensorRT::AddSoftMaxOp(TensorRTContext *ctx) {
  nvinfer1::ISoftMaxLayer *current_layer_ = ctx->network()->addSoftMax(*input(ctx, 0).trt_tensor_);
  if (current_layer_ == nullptr) {
    MS_LOG(ERROR) << "add softmax op failed for TensorRT.";
    return nullptr;
  }

  int64_t axis_format_value =
    (axis_val_[0] == -1) ? input(ctx, 0).trt_tensor_->getDimensions().nbDims - 1 : axis_val_[0];
  uint32_t axis_bit = 1 << axis_format_value;
  MS_LOG(DEBUG) << op_name_ << " axis_value is " << axis_format_value << ", set axis to " << axis_bit;
  current_layer_->setAxes(axis_bit);
  return current_layer_;
}
REGISTER_TENSORRT_CREATOR(ops::kNameSoftmax, SoftMaxTensorRT)
REGISTER_TENSORRT_CREATOR(ops::kNameLogSoftmax, SoftMaxTensorRT)
}  // namespace mindspore::lite
