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

#include "src/extendrt/delegate/tensorrt/op/square_tensorrt.h"

namespace mindspore::lite {
int SquareTensorRT::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                              const std::vector<mindspore::MSTensor> &out_tensors) {
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

int SquareTensorRT::AddInnerOp(TensorRTContext *ctx) {
  if (ctx == nullptr || ctx->network() == nullptr) {
    MS_LOG(ERROR) << "context or network is invalid";
    return RET_ERROR;
  }
  auto square_op = op_primitive_->value_as_Square();
  CHECK_NULL_RETURN(square_op);
  int input_nbdims = input(ctx, 0).trt_tensor_->getDimensions().nbDims;
  if (input_nbdims == -1) {
    MS_LOG(ERROR) << "square failed for " << op_name_;
    return RET_ERROR;
  }
  int ret = RunAsTrtOps(ctx);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "square failed for " << op_name_;
    return ret;
  }
  return ret;
}

int SquareTensorRT::RunAsTrtOps(TensorRTContext *ctx) {
  if (ctx == nullptr || ctx->network() == nullptr) {
    MS_LOG(ERROR) << "context or network is invalid";
    return RET_ERROR;
  }
  auto square_layer = ctx->network()->addElementWise(*input(ctx, 0).trt_tensor_, *input(ctx, 0).trt_tensor_,
                                                     nvinfer1::ElementWiseOperation::kPROD);
  CHECK_NULL_RETURN(square_layer);
  auto out_tensor = square_layer->getOutput(0);
  CHECK_NULL_RETURN(out_tensor);
  this->layer_ = square_layer;
  ctx->RegisterTensor(ITensorHelper{out_tensor, input(ctx, 0).format_, input(ctx, 0).same_format_},
                      out_tensors_[0].Name());
  return RET_OK;
}
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_Square, SquareTensorRT)
}  // namespace mindspore::lite
