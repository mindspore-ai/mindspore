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

#include "src/litert/delegate/tensorrt/op/batchnorm_tensorrt.h"
#include <memory>
#include <numeric>

namespace mindspore::lite {
int BatchNormTensorRT::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                                 const std::vector<mindspore::MSTensor> &out_tensors) {
  if (in_tensors.size() != INPUT_SIZE5 && in_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  return RET_OK;
}

int BatchNormTensorRT::AddInnerOp(TensorRTContext *ctx) {
  CHECK_NULL_RETURN(ctx->network());
  auto norm_op = op_primitive_->value_as_FusedBatchNorm();
  CHECK_NULL_RETURN(norm_op);
  epsilon_ = norm_op->epsilon();

  ITensorHelper norm_input;
  int ret = PreprocessInputs2SameDim(ctx, input(ctx, 0), &norm_input);
  if (ret != RET_OK || norm_input.trt_tensor_ == nullptr) {
    MS_LOG(ERROR) << "PreprocessInputs2SameDim norm_input failed for " << op_name_;
    return RET_ERROR;
  }
  auto expect_shape = ConvertMSShape(norm_input.trt_tensor_->getDimensions());
  gamma_ = ConvertTensorWithExpandDims(ctx, in_tensors_[1], expect_shape, op_name_ + in_tensors_[1].Name());
  CHECK_NULL_RETURN(gamma_);
  beta_ =
    ConvertTensorWithExpandDims(ctx, in_tensors_[BETA_INDEX], expect_shape, op_name_ + in_tensors_[BETA_INDEX].Name());
  CHECK_NULL_RETURN(beta_);
  mean_ =
    ConvertTensorWithExpandDims(ctx, in_tensors_[MEAN_INDEX], expect_shape, op_name_ + in_tensors_[MEAN_INDEX].Name());
  CHECK_NULL_RETURN(mean_);
  var_ =
    ConvertTensorWithExpandDims(ctx, in_tensors_[VAR_INDEX], expect_shape, op_name_ + in_tensors_[VAR_INDEX].Name());
  CHECK_NULL_RETURN(var_);

  return RunAsTrtOps(ctx, norm_input);
}

int BatchNormTensorRT::RunAsTrtOps(TensorRTContext *ctx, ITensorHelper norm_input) {
  // var + min epsilon
  auto const_epsilon = ConvertScalarToITensor(ctx, norm_input.trt_tensor_->getDimensions().nbDims, &epsilon_,
                                              DataType::kNumberTypeFloat32, op_name_ + "_epsilion");
  CHECK_NULL_RETURN(const_epsilon);
  auto var_epsilon =
    ctx->network()->addElementWise(*var_, *const_epsilon, nvinfer1::ElementWiseOperation::kSUM)->getOutput(0);
  CHECK_NULL_RETURN(var_epsilon);

  // standard deviation
  auto std_dev = ctx->network()->addUnary(*var_epsilon, nvinfer1::UnaryOperation::kSQRT)->getOutput(0);
  CHECK_NULL_RETURN(std_dev);

  auto scale = ctx->network()->addElementWise(*gamma_, *std_dev, nvinfer1::ElementWiseOperation::kDIV)->getOutput(0);
  CHECK_NULL_RETURN(scale);

  auto mean_mul_scale =
    ctx->network()->addElementWise(*mean_, *scale, nvinfer1::ElementWiseOperation::kPROD)->getOutput(0);
  CHECK_NULL_RETURN(mean_mul_scale);

  auto bias =
    ctx->network()->addElementWise(*beta_, *mean_mul_scale, nvinfer1::ElementWiseOperation::kSUB)->getOutput(0);
  CHECK_NULL_RETURN(bias);

  // scale with bias
  auto scale_layer =
    ctx->network()->addElementWise(*norm_input.trt_tensor_, *scale, nvinfer1::ElementWiseOperation::kPROD);
  this->layer_ = scale_layer;
  auto scale_out = scale_layer->getOutput(0);
  CHECK_NULL_RETURN(scale_out);
  auto beta_out = ctx->network()->addElementWise(*scale_out, *bias, nvinfer1::ElementWiseOperation::kSUM)->getOutput(0);
  CHECK_NULL_RETURN(beta_out);
  ctx->RegisterTensor(ITensorHelper{beta_out, Format::NHWC, true}, out_tensors_[0].Name());
  return RET_OK;
}
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_FusedBatchNorm, BatchNormTensorRT)
}  // namespace mindspore::lite
