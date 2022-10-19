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

#include <memory>
#include "src/extendrt/delegate/tensorrt/op/rsqrt_tensorrt.h"
#include "ops/rsqrt.h"

namespace mindspore::lite {
int RsqrtTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                             const std::vector<TensorInfo> &out_tensors) {
  if (in_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  return RET_OK;
}

int RsqrtTensorRT::AddInnerOp(TensorRTContext *ctx) {
  if (ctx == nullptr || ctx->network() == nullptr) {
    MS_LOG(ERROR) << "context or network is invalid for " << op_name_;
    return RET_ERROR;
  }
  int input_nbdims = input(ctx, 0).trt_tensor_->getDimensions().nbDims;
  if (input_nbdims == -1) {
    MS_LOG(ERROR) << "Invalid input dims " << input_nbdims << " for " << op_name_;
    return RET_ERROR;
  }
  int ret = RunAsTrtOps(ctx);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Rsqrt op failed for " << op_name_;
    return ret;
  }
  return ret;
}

int RsqrtTensorRT::RunAsTrtOps(TensorRTContext *ctx) {
  if (ctx == nullptr || ctx->network() == nullptr) {
    MS_LOG(ERROR) << "context or network is invalid for " << op_name_;
    return RET_ERROR;
  }
  auto const_one = ctx->ConvertTo1DTensor(std::vector<float>(input(ctx, 0).trt_tensor_->getDimensions().nbDims, 1.f));
  CHECK_NULL_RETURN(const_one);
  auto sqrt_tensor =
    ctx->network()->addUnary(*input(ctx, 0).trt_tensor_, nvinfer1::UnaryOperation::kSQRT)->getOutput(0);
  auto rsqrt_layer = ctx->network()->addElementWise(*const_one, *sqrt_tensor, nvinfer1::ElementWiseOperation::kDIV);
  CHECK_NULL_RETURN(rsqrt_layer);
  auto out_tensor = rsqrt_layer->getOutput(0);
  CHECK_NULL_RETURN(out_tensor);
  this->layer_ = rsqrt_layer;
  ctx->RegisterTensor(ITensorHelper{out_tensor, input(ctx, 0).format_, input(ctx, 0).same_format_},
                      out_tensors_[0].Name());
  return RET_OK;
}
REGISTER_TENSORRT_CREATOR(ops::kNameRsqrt, RsqrtTensorRT)
}  // namespace mindspore::lite
