/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "src/extendrt/delegate/tensorrt/op/prelu_tensorrt.h"
#include <numeric>
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"
#include "ops/fusion/prelu_fusion.h"

namespace mindspore::lite {
int PReluTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                             const std::vector<TensorInfo> &out_tensors) {
  if (!IsShapeKnown()) {
    MS_LOG(ERROR) << "Unsupported input tensor unknown shape: " << op_name_;
    return RET_ERROR;
  }
  if (in_tensors.size() != INPUT_SIZE2) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size() << " : " << op_name_;
    return RET_ERROR;
  }

  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size() << " : " << op_name_;
    return RET_ERROR;
  }
  return RET_OK;
}

int PReluTensorRT::AddInnerOp(TensorRTContext *ctx) {
  ITensorHelper prelu_input;
  int ret = PreprocessInputs2SameDim(ctx, input(ctx, 0), &prelu_input);
  if (ret != RET_OK || prelu_input.trt_tensor_ == nullptr) {
    MS_LOG(ERROR) << "PreprocessInputs2SameDim input tensor failed for " << op_name_;
    return ret;
  }
  int input_nbdims = prelu_input.trt_tensor_->getDimensions().nbDims;
  int slope_nbdims = in_tensors_[1].Shape().size();
  ITensorHelper slope_helper;
  if (input_nbdims != slope_nbdims) {
    auto expect_shape = ConvertMSShape(input(ctx, 0).trt_tensor_->getDimensions());
    slope_helper.trt_tensor_ = ConvertTensorWithExpandDims(ctx, in_tensors_[1], expect_shape, op_name_ + "_slope");
  }
  if (slope_helper.trt_tensor_ == nullptr) {
    MS_LOG(ERROR) << "add const input tensor failed for " << op_name_;
    return RET_ERROR;
  }
  ret = PreprocessInputs2SameDim(ctx, slope_helper, &slope_helper);
  if (ret != RET_OK || slope_helper.trt_tensor_ == nullptr) {
    MS_LOG(ERROR) << "PreprocessInputs2SameDim slope tensor failed for " << op_name_;
    return ret;
  }

  auto *prelu_layer = ctx->network()->addParametricReLU(*prelu_input.trt_tensor_, *slope_helper.trt_tensor_);
  if (prelu_layer == nullptr) {
    MS_LOG(ERROR) << "addParameticReLU failed for TensorRT : " << op_name_;
    return RET_ERROR;
  }

  nvinfer1::ITensor *out_tensor = prelu_layer->getOutput(0);
  ctx->RegisterTensor(ITensorHelper{out_tensor, prelu_input.format_, prelu_input.same_format_}, out_tensors_[0].Name());
  this->layer_ = prelu_layer;
  return RET_OK;
}
REGISTER_TENSORRT_CREATOR(ops::kNamePReLUFusion, PReluTensorRT)
}  // namespace mindspore::lite
