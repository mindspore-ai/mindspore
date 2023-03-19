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

#include "src/extendrt/delegate/tensorrt/op/gather_tensorrt.h"
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"
#include "ops/gather.h"

namespace mindspore::lite {
constexpr int AXIS_INDEX = 2;

int GatherTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                              const std::vector<TensorInfo> &out_tensors) {
  if (in_tensors.size() != INPUT_SIZE3) {
    MS_LOG(ERROR) << "invalid input tensor size: " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "invalid output tensor size: " << out_tensors.size();
    return RET_ERROR;
  }
  if (in_tensors[1].DataType() != DataType::kNumberTypeInt32 &&
      in_tensors[1].DataType() != DataType::kNumberTypeInt64) {
    MS_LOG(ERROR) << "Gather indices only support Int32";
    return RET_ERROR;
  }
  if (in_tensors[AXIS_INDEX].ElementNum() == 1) {
    auto axis_vec = ConvertTensorAsIntVector(in_tensors_[AXIS_INDEX]);
    if (axis_vec.size() != 1) {
      MS_LOG(ERROR) << "Failed to get axis input, dim count " << axis_vec.size() << ", node: " << op_name_;
      return RET_ERROR;
    }
    axis_ = axis_vec[0];
  } else {
    MS_LOG(ERROR) << "TensorRT axis is attribute.";
    return RET_ERROR;
  }
  return RET_OK;
}

int GatherTensorRT::AddInnerOp(TensorRTContext *ctx) {
  if (ctx == nullptr || ctx->network() == nullptr) {
    MS_LOG(ERROR) << "context or network is invalid";
    return RET_ERROR;
  }
  for (size_t i = 0; i != AXIS_INDEX; ++i) {
    if (input(ctx, i).trt_tensor_ == nullptr) {
      auto const_input = ConvertConstantTensor(ctx, in_tensors_[i], op_name_);
      auto is_scalar = in_tensors_[i].Shape().empty();
      ctx->RegisterTensor(ITensorHelper{const_input, NCHW, true, !is_scalar}, in_tensors_[i].Name());
    }
  }

  ITensorHelper gather_input = input(ctx, 0);
  int ret = PreprocessInputs2SameDim(ctx, gather_input, &gather_input);
  if (ret != RET_OK || gather_input.trt_tensor_ == nullptr) {
    MS_LOG(ERROR) << "PreprocessInputs2SameDim gather failed for " << op_name_;
    return RET_ERROR;
  }
  ITensorHelper indices_tensor = input(ctx, 1);
  ret = PreprocessInputs2SameDim(ctx, indices_tensor, &indices_tensor);
  if (ret != RET_OK || indices_tensor.trt_tensor_ == nullptr) {
    MS_LOG(ERROR) << "PreprocessInputs2SameDim indices failed for " << op_name_;
    return RET_ERROR;
  }

  nvinfer1::IGatherLayer *gather_layer =
    ctx->network()->addGather(*gather_input.trt_tensor_, *indices_tensor.trt_tensor_, axis_);
  if (gather_layer == nullptr) {
    MS_LOG(ERROR) << "addGather failed for TensorRT.";
    return RET_ERROR;
  }

  this->layer_ = gather_layer;
  gather_layer->setName(op_name_.c_str());
  nvinfer1::ITensor *op_output = gather_layer->getOutput(0);
  auto old_shape = ConvertMSShape(op_output->getDimensions());
  // keep shape
  if (!indices_tensor.is_tensor && old_shape.size() > 1) {
    auto squeeze = ctx->network()->addShuffle(*op_output);
    if (squeeze == nullptr) {
      MS_LOG(ERROR) << "add output squeeze failed for " << op_name_;
      return RET_ERROR;
    }
    squeeze->setName((op_name_ + "_squeeze_out").c_str());
    old_shape.erase(old_shape.begin() + axis_);
    squeeze->setReshapeDimensions(ConvertCudaDims(old_shape));
    op_output = squeeze->getOutput(0);
  }

  auto out_helper = ITensorHelper{op_output, gather_input.format_, gather_input.same_format_};
  if (old_shape.size() == 1) {
    out_helper.is_tensor = false;
  }
  ctx->RegisterTensor(out_helper, out_tensors_[0].Name());
  return RET_OK;
}
REGISTER_TENSORRT_CREATOR(ops::kNameGather, GatherTensorRT)
}  // namespace mindspore::lite
