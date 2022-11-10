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

#include "src/extendrt/delegate/tensorrt/op/l2normalization_tensorrt.h"
#include <numeric>
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"
#include "ops/fusion/l2_normalize_fusion.h"

namespace mindspore::lite {
int L2NormalizationTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                                       const std::vector<TensorInfo> &out_tensors) {
  if (in_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size() << " : " << op_name_;
    return RET_ERROR;
  }

  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size() << " : " << op_name_;
    return RET_ERROR;
  }
  return RET_OK;
}

int L2NormalizationTensorRT::AddInnerOp(TensorRTContext *ctx) {
  auto input_tensor = input(ctx, 0).trt_tensor_;
  int nbdims = input_tensor->getDimensions().nbDims;
  auto op = AsOps<ops::L2NormalizeFusion>();
  int64_t axis = op->get_axis()[0];
  if (axis < 0) {
    axis += nbdims;
  }

  if (axis < 0 || axis >= nbdims) {
    MS_LOG(ERROR) << "axis error : " << axis << " for " << op_name_;
    return RET_ERROR;
  }

  float epsilon = op->get_epsilon();

  auto pow =
    ctx->network()->addElementWise(*input_tensor, *input_tensor, nvinfer1::ElementWiseOperation::kPROD)->getOutput(0);
  auto sum = ctx->network()->addReduce(*pow, nvinfer1::ReduceOperation::kSUM, 1 << axis, true)->getOutput(0);

  auto ep = ctx->ConvertTo1DTensor(epsilon);
  while (ep->getDimensions().nbDims < nbdims) {
    ep = ExpandDim(ctx, ep, 0);
  }

  if (input_tensor->getType() != nvinfer1::DataType::kFLOAT) {
    ep = TRTTensorCast(ctx, ep, input_tensor->getType(), op_name_ + "_cast_epsilon");
  }

  auto norm = ctx->network()->addElementWise(*sum, *ep, nvinfer1::ElementWiseOperation::kMAX)->getOutput(0);
  norm = ctx->network()->addUnary(*norm, nvinfer1::UnaryOperation::kSQRT)->getOutput(0);
  auto div_layer = ctx->network()->addElementWise(*input_tensor, *norm, nvinfer1::ElementWiseOperation::kDIV);

  nvinfer1::ITensor *out_tensor = div_layer->getOutput(0);
  ctx->RegisterTensor(ITensorHelper{out_tensor, Format::NCHW, true}, out_tensors_[0].Name());
  this->layer_ = div_layer;
  return RET_OK;
}
REGISTER_TENSORRT_CREATOR(ops::kNameL2NormalizeFusion, L2NormalizationTensorRT)
}  // namespace mindspore::lite
