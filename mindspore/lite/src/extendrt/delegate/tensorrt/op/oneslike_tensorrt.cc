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
#include <functional>
#include "src/extendrt/delegate/tensorrt/op/oneslike_tensorrt.h"
#include "ops/ones_like.h"

namespace mindspore::lite {
int OneslikeTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
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

int OneslikeTensorRT::AddInnerOp(TensorRTContext *ctx) {
  if (ctx == nullptr || ctx->network() == nullptr) {
    MS_LOG(ERROR) << "context or network is invalid";
    return RET_ERROR;
  }
  int ret = RunAsTrtOps(ctx);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "oneslike op failed for " << op_name_;
    return ret;
  }
  return ret;
}

int OneslikeTensorRT::RunAsTrtOps(TensorRTContext *ctx) {
  if (ctx == nullptr || ctx->network() == nullptr) {
    MS_LOG(ERROR) << "context or network is invalid";
    return RET_ERROR;
  }
  auto input_trt_tensor = input(ctx, 0).trt_tensor_;
  nvinfer1::ITensor *value_tensor;
  if (in_tensors_[0].DataType() == DataType::kNumberTypeFloat32) {
    const float value = 1.f;
    value_tensor = ctx->ConvertTo1DTensor(value);
  } else if (in_tensors_[0].DataType() == DataType::kNumberTypeInt32) {
    const int value = 1;
    value_tensor = ctx->ConvertTo1DTensor(value);
  } else {
    MS_LOG(ERROR) << "dtype not implement: " << in_tensors_[0].DataType();
    return RET_ERROR;
  }
  CHECK_NULL_RETURN(value_tensor);
  auto unsqueeze_layer = ctx->network()->addShuffle(*value_tensor);
  CHECK_NULL_RETURN(unsqueeze_layer);

  auto shape_tensor = ctx->network()->addShape(*input_trt_tensor)->getOutput(0);
  CHECK_NULL_RETURN(shape_tensor);
  int rank = shape_tensor->getDimensions().d[0];
  nvinfer1::Dims unsqueeze{rank};
  std::fill(unsqueeze.d, unsqueeze.d + rank, 1);
  unsqueeze_layer->setReshapeDimensions(unsqueeze);
  unsqueeze_layer->setZeroIsPlaceholder(false);
  value_tensor = unsqueeze_layer->getOutput(0);
  CHECK_NULL_RETURN(value_tensor);

  auto out_tensor = Broadcast(ctx, value_tensor, shape_tensor);

  CHECK_NULL_RETURN(out_tensor);
  ctx->RegisterTensor(ITensorHelper{out_tensor, input(ctx, 0).format_, input(ctx, 0).same_format_},
                      out_tensors_[0].Name());
  return RET_OK;
}
REGISTER_TENSORRT_CREATOR(ops::kNameOnesLike, OneslikeTensorRT)
}  // namespace mindspore::lite
