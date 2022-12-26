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
#include "src/extendrt/delegate/tensorrt/op/fastgelu_tensorrt.h"
#include "src/extendrt/delegate/tensorrt/op/cast_tensorrt.h"
#include "ops/fast_gelu.h"
#include "src/extendrt/delegate/tensorrt/tensorrt_utils.h"

namespace mindspore::lite {
// FastGelu is defined as "0.5x * (1 + tanh[sqrt(2.0/Pi) * (x + 0.044715 * x^3)])" in paper "Gaussian Error Linear Units
// (GELUs)" by Dan Hendrycks, Kevin Gimpel, 2016.
constexpr float FASTGELU_PARAM1 = 3.f;
constexpr float FASTGELU_PARAM2 = 0.044715f;
constexpr float FASTGELU_PARAM3 = 0.7978845608f;  // sqrt(2.0/Pi)
constexpr float FASTGELU_PARAM4 = 1.f;
constexpr float FASTGELU_PARAM5 = 0.5f;

int FastGeluTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
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

int FastGeluTensorRT::AddInnerOp(TensorRTContext *ctx) {
  if (ctx == nullptr || ctx->network() == nullptr) {
    MS_LOG(ERROR) << "context or network is invalid";
    return RET_ERROR;
  }
  int input_nbdims = input(ctx, 0).trt_tensor_->getDimensions().nbDims;
  if (input_nbdims < 0) {
    MS_LOG(ERROR) << "input dims should not be less than 0 for " << op_name_;
    return RET_ERROR;
  }
  int ret = RunAsTrtOps(ctx);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "add layer failed for " << op_name_;
    return ret;
  }
  return ret;
}

int FastGeluTensorRT::RunAsTrtOps(TensorRTContext *ctx) {
  if (ctx == nullptr || ctx->network() == nullptr) {
    MS_LOG(ERROR) << "context or network is invalid for " << op_name_;
    return RET_ERROR;
  }
  auto trt_in_tensor = input(ctx, 0).trt_tensor_;
  if (trt_in_tensor->getDimensions().nbDims <= 0) {
    MS_LOG(ERROR) << "Invalid input dims count " << trt_in_tensor->getDimensions().nbDims << ", " << op_name_;
    return RET_ERROR;
  }
  auto expand_dims = [](TensorRTContext *ctx, nvinfer1::ITensor *tensor, int nbdims) {
    while (tensor->getDimensions().nbDims != nbdims) {
      tensor = ExpandDim(ctx, tensor, 0);
    }
    return tensor;
  };
  int nbdims = trt_in_tensor->getDimensions().nbDims;
  auto const_three = expand_dims(ctx, ctx->ConvertTo1DTensor(FASTGELU_PARAM1), nbdims);
  CHECK_NULL_RETURN(const_three);
  auto p3 =
    ctx->network()->addElementWise(*trt_in_tensor, *const_three, nvinfer1::ElementWiseOperation::kPOW)->getOutput(0);
  CHECK_NULL_RETURN(p3);
  auto gelu_p1 = expand_dims(ctx, ctx->ConvertTo1DTensor(FASTGELU_PARAM2), nbdims);
  CHECK_NULL_RETURN(gelu_p1);
  auto prod1 = ctx->network()->addElementWise(*p3, *gelu_p1, nvinfer1::ElementWiseOperation::kPROD)->getOutput(0);
  CHECK_NULL_RETURN(prod1);
  auto sum = ctx->network()->addElementWise(*prod1, *trt_in_tensor, nvinfer1::ElementWiseOperation::kSUM)->getOutput(0);
  CHECK_NULL_RETURN(sum);
  auto gelu_p2 = expand_dims(ctx, ctx->ConvertTo1DTensor(FASTGELU_PARAM3), nbdims);
  CHECK_NULL_RETURN(gelu_p2);
  auto prod2 = ctx->network()->addElementWise(*sum, *gelu_p2, nvinfer1::ElementWiseOperation::kPROD)->getOutput(0);
  CHECK_NULL_RETURN(prod2);
  auto tanh = ctx->network()->addActivation(*prod2, nvinfer1::ActivationType::kTANH)->getOutput(0);
  CHECK_NULL_RETURN(tanh);
  auto const_one = expand_dims(ctx, ctx->ConvertTo1DTensor(FASTGELU_PARAM4), nbdims);
  CHECK_NULL_RETURN(const_one);
  auto sum2 = ctx->network()->addElementWise(*const_one, *tanh, nvinfer1::ElementWiseOperation::kSUM)->getOutput(0);
  CHECK_NULL_RETURN(sum2);
  auto prod3 =
    ctx->network()->addElementWise(*sum2, *trt_in_tensor, nvinfer1::ElementWiseOperation::kPROD)->getOutput(0);
  CHECK_NULL_RETURN(prod3);
  auto gelu_p3 = expand_dims(ctx, ctx->ConvertTo1DTensor(FASTGELU_PARAM5), nbdims);
  CHECK_NULL_RETURN(gelu_p3);
  auto fastgelu_layer = ctx->network()->addElementWise(*prod3, *gelu_p3, nvinfer1::ElementWiseOperation::kPROD);
  if (fastgelu_layer == nullptr) {
    MS_LOG(ERROR) << "add fastgelu op failed for TensorRT.";
    return RET_ERROR;
  }
  nvinfer1::ITensor *out_tensor = fastgelu_layer->getOutput(0);
  if (out_tensor == nullptr) {
    MS_LOG(ERROR) << "add fastgelu op failed for TensorRT.";
    return RET_ERROR;
  }

  // cast to origin type
  if (out_tensor->getType() != ConvertDataType(out_tensors_[0].DataType())) {
    out_tensor = TRTTensorCast(ctx, fastgelu_layer->getOutput(0), ConvertDataType(out_tensors_[0].DataType()),
                               op_name_ + "_cast_out");
  }
  ctx->RegisterTensor(ITensorHelper{out_tensor, input(ctx, 0).format_, input(ctx, 0).same_format_},
                      out_tensors_[0].Name());
  fastgelu_layer->setName(op_name_.c_str());
  this->layer_ = fastgelu_layer;
  ctx->RegisterLayer(fastgelu_layer, op_name_);
  return RET_OK;
}
REGISTER_TENSORRT_CREATOR(ops::kNameFastGeLU, FastGeluTensorRT)
}  // namespace mindspore::lite
