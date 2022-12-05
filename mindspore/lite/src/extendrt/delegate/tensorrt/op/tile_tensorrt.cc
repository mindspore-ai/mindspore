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

#include "src/extendrt/delegate/tensorrt/op/tile_tensorrt.h"
#include <cuda_runtime.h>
#include <numeric>
#include <memory>
#include <functional>
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"
#include "ops/fusion/tile_fusion.h"

namespace mindspore::lite {
int TileTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                            const std::vector<TensorInfo> &out_tensors) {
  if (in_tensors.size() != INPUT_SIZE2) {
    MS_LOG(ERROR) << "invalid input tensor size: " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "invalid output tensor size: " << out_tensors.size();
    return RET_ERROR;
  }
  return RET_OK;
}

int TileTensorRT::AddInnerOp(TensorRTContext *ctx) {
  auto repeats_tensor = in_tensors_[1];
  ITensorHelper tile_input = input(ctx, 0);
  nvinfer1::ITensor *output;
  auto input_shape = ctx->network()->addShape(*input(ctx, 0).trt_tensor_)->getOutput(0);
  if (repeats_tensor.IsConst()) {
    if (repeats_tensor.ElementNum() != input(ctx, 0).trt_tensor_->getDimensions().nbDims) {
      MS_LOG(ERROR) << op_name_ << " has input dims: " << input(ctx, 0).trt_tensor_->getDimensions().nbDims
                    << ", and invalid repeats cnt: " << repeats_tensor.ElementNum();
      return RET_ERROR;
    }
    auto ret = ParseData2Vector(in_tensors_[1], &repeats_);
    if (ret != RET_OK || repeats_.size() == 0) {
      MS_LOG(ERROR) << op_name_ << " has invalid repeats tensor";
      return ret;
    }
    std::vector<int> repeats(repeats_.size());
    for (size_t i = 0; i != repeats_.size(); ++i) {
      repeats[i] = static_cast<int>(repeats_[i]);
    }
    auto repeat_tensor = ctx->ConvertTo1DTensor(repeats);
    auto output_shape =
      ctx->network()->addElementWise(*input_shape, *repeat_tensor, nvinfer1::ElementWiseOperation::kPROD)->getOutput(0);
    output = Broadcast(ctx, tile_input.trt_tensor_, output_shape);
  } else {
    auto output_shape =
      ctx->network()
        ->addElementWise(*input_shape, *input(ctx, 1).trt_tensor_, nvinfer1::ElementWiseOperation::kPROD)
        ->getOutput(0);
    output = Broadcast(ctx, tile_input.trt_tensor_, output_shape);
  }
  auto layer = ctx->network()->addIdentity(*output);
  layer_ = layer;
  auto tile_out = layer->getOutput(0);
  ctx->RegisterTensor(ITensorHelper{tile_out, tile_input.format_, true}, out_tensors_[0].Name());
  return RET_OK;
}
REGISTER_TENSORRT_CREATOR(ops::kNameTileFusion, TileTensorRT)
}  // namespace mindspore::lite
