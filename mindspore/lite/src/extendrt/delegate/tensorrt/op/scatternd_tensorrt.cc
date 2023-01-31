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

#include "src/extendrt/delegate/tensorrt/op/scatternd_tensorrt.h"
#include <numeric>
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"
#include "ops/scatter_nd_update.h"
#include "ops/tensor_scatter_update.h"
#include "ops/tensor_scatter_add.h"

namespace mindspore::lite {
int ScatterNdTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                                 const std::vector<TensorInfo> &out_tensors) {
#if TRT_VERSION_GE(8, 2)
  if (in_tensors.size() != INPUT_SIZE3) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size() << " : " << op_name_;
    return RET_ERROR;
  }

  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size() << " : " << op_name_;
    return RET_ERROR;
  }
  return RET_OK;
#else
  MS_LOG(WARNING) << "low TensorRT version don't support Scatter op, please upgrade TensorRT version to 8.2 or higher";
  return RET_ERROR;
#endif
}

int ScatterNdTensorRT::AddInnerOp(TensorRTContext *ctx) {
#if TRT_VERSION_GE(8, 2)
  ITensorHelper scatter_input = input(ctx, 0);
  if (in_tensors_[0].IsConst() && scatter_input.trt_tensor_ == nullptr) {
    scatter_input.trt_tensor_ = lite::ConvertConstantTensor(ctx, in_tensors_[0], op_name_);
    scatter_input.format_ = Format::NCHW;
    ctx->RegisterTensor(scatter_input, in_tensors_[0].Name());
  }
  ITensorHelper indices_helper;
  int ret = PreprocessInputs2SameDim(ctx, input(ctx, 1), &indices_helper);
  if (ret != RET_OK || indices_helper.trt_tensor_ == nullptr) {
    MS_LOG(ERROR) << "PreprocessInputs2SameDim indices tensor failed for " << op_name_;
    return ret;
  }
  ITensorHelper updates_helper;
  ret = PreprocessInputs2SameDim(ctx, input(ctx, INPUT_SIZE2), &updates_helper);
  if (ret != RET_OK || updates_helper.trt_tensor_ == nullptr) {
    MS_LOG(ERROR) << "PreprocessInputs2SameDim update tensor failed for " << op_name_;
    return ret;
  }

  nvinfer1::IScatterLayer *scatter_layer = ctx->network()->addScatter(
    *scatter_input.trt_tensor_, *indices_helper.trt_tensor_, *updates_helper.trt_tensor_, nvinfer1::ScatterMode::kND);
  if (scatter_layer == nullptr) {
    MS_LOG(ERROR) << "addScatter failed for TensorRT.";
    return RET_ERROR;
  }

  nvinfer1::ITensor *out_tensor = scatter_layer->getOutput(0);
  ctx->RegisterTensor(ITensorHelper{out_tensor, scatter_input.format_, scatter_input.same_format_},
                      out_tensors_[0].Name());
  this->layer_ = scatter_layer;
  return RET_OK;
#else
  MS_LOG(WARNING) << "low TensorRT version don't support Scatter op, please upgrade TensorRT version to 8.2 or higher";
  return RET_ERROR;
#endif
}
REGISTER_TENSORRT_CREATOR(ops::kNameScatterNdUpdate, ScatterNdTensorRT)
REGISTER_TENSORRT_CREATOR(ops::kNameTensorScatterUpdate, ScatterNdTensorRT)
}  // namespace mindspore::lite
