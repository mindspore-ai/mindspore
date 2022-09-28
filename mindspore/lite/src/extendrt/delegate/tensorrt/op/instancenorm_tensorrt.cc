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

#include "src/extendrt/delegate/tensorrt/op/instancenorm_tensorrt.h"
#include <memory>
#include <numeric>
#include "ops/instance_norm.h"

namespace mindspore::lite {
namespace {
constexpr int GAMMA_INDEX = 1;
constexpr int BETA_INDEX = 2;
}  // namespace
int InstanceNormTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                                    const std::vector<TensorInfo> &out_tensors) {
  if (in_tensors.size() != INPUT_SIZE3) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  if (!in_tensors_[GAMMA_INDEX].IsConst() || !in_tensors_[BETA_INDEX].IsConst()) {
    MS_LOG(ERROR) << "Unsupported non const gamma or beta input, is gamma const: " << in_tensors_[GAMMA_INDEX].IsConst()
                  << ", is beta const: " << in_tensors_[BETA_INDEX].IsConst();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
    return RET_ERROR;
  }
  dynamic_shape_params_.support_dynamic_ = false;
  dynamic_shape_params_.support_hw_dynamic_ = false;
  return RET_OK;
}

int InstanceNormTensorRT::AddInnerOp(TensorRTContext *ctx) {
  CHECK_NULL_RETURN(ctx->network());
  auto norm_op = AsOps<ops::InstanceNorm>();
  CHECK_NULL_RETURN(norm_op);
  epsilon_ = norm_op->get_epsilon();

  ITensorHelper norm_input = input(ctx, 0);
  if (norm_input.trt_tensor_ == nullptr) {
    MS_LOG(ERROR) << "Input tensorrt tensor cannot be nullptr, op: " << op_name_;
    return RET_ERROR;
  }
  auto norm_input_dims = norm_input.trt_tensor_->getDimensions();
  if (norm_input_dims.nbDims != kDim4) {
    MS_LOG(ERROR) << "Expect count of input dims to be " << kDim4 << ", but got: " << CudaDimsAsString(norm_input_dims)
                  << " , op: " << op_name_;
    return RET_ERROR;
  }
  if (IsDynamicInput(ctx, 0)) {
    MS_LOG(ERROR) << "Not support dynamic input, input dims: " << CudaDimsAsString(norm_input_dims)
                  << ", op: " << op_name_;
    return RET_ERROR;
  }
  auto &gamma_input = in_tensors_[GAMMA_INDEX];
  auto &beta_input = in_tensors_[BETA_INDEX];
  auto nc = norm_input_dims.d[0] * norm_input_dims.d[1];
  if (gamma_input.ElementNum() != nc || beta_input.ElementNum() != nc) {
    MS_LOG(ERROR) << "Element number of gamma or beta expect to be N*C of input, but got gamma element number: "
                  << gamma_input.ElementNum() << ", beta element number: " << beta_input.ElementNum()
                  << ", input dims: " << CudaDimsAsString(norm_input_dims) << ", op: " << op_name_;
    return RET_ERROR;
  }
  auto expect_shape = ConvertMSShape(norm_input_dims);
  if (gamma_input.Shape().size() == 1) {
    expect_shape[kDim2] = expect_shape[kDim2] * expect_shape[kDim3];
    expect_shape.erase(expect_shape.begin() + kDim3);
  }
  gamma_ = ConvertTensorWithExpandDims(ctx, gamma_input, expect_shape, op_name_ + gamma_input.Name());
  CHECK_NULL_RETURN(gamma_);
  beta_ = ConvertTensorWithExpandDims(ctx, beta_input, expect_shape, op_name_ + beta_input.Name());
  CHECK_NULL_RETURN(beta_);

  auto reshape_layer = ctx->network()->addShuffle(*norm_input.trt_tensor_);
  auto reshape_shape = ConvertMSShape(norm_input_dims);
  reshape_shape[kDim2] = reshape_shape[kDim2] * reshape_shape[kDim3];
  reshape_shape.erase(reshape_shape.begin() + kDim3);
  reshape_layer->setReshapeDimensions(ConvertCudaDims(reshape_shape));
  // n,c,hw
  auto reshape_output = reshape_layer->getOutput(0);

  constexpr uint32_t reduce_axis_hw = (1 << 2);
  // scale = gama / sqrt(mean(hw*hw) - mean(hw)^2 + epsilon_)
  // dst[index] = (src[index] - mean(hw)) * scale + beta_data = src[index]*scale - mean(hw)*scale + beta_data
  // mean(hw)
  auto mean_layer = ctx->network()->addReduce(*reshape_output, nvinfer1::ReduceOperation::kAVG, reduce_axis_hw, true);
  auto mean_output = mean_layer->getOutput(0);
  // mean(hw)^2
  auto mean_square_layer =
    ctx->network()->addElementWise(*mean_output, *mean_output, nvinfer1::ElementWiseOperation::kPROD);
  auto mean_square_output = mean_square_layer->getOutput(0);
  // hw*hw
  auto square_layer =
    ctx->network()->addElementWise(*reshape_output, *reshape_output, nvinfer1::ElementWiseOperation::kPROD);
  auto square_output = square_layer->getOutput(0);
  // mean(hw*hw)
  auto square_mean_layer =
    ctx->network()->addReduce(*square_output, nvinfer1::ReduceOperation::kAVG, reduce_axis_hw, true);
  auto square_mean_output = square_mean_layer->getOutput(0);
  // mean(hw*hw) - mean(hw)^2
  auto var_layer =
    ctx->network()->addElementWise(*square_mean_output, *mean_square_output, nvinfer1::ElementWiseOperation::kSUB);
  auto var_output = var_layer->getOutput(0);

  auto const_epsilon = ConvertScalarToITensor(ctx, var_output->getDimensions().nbDims, &epsilon_,
                                              DataType::kNumberTypeFloat32, op_name_ + "_epsilion");
  CHECK_NULL_RETURN(const_epsilon);
  // mean(hw*hw) - mean(hw)^2 + epsilon_
  auto var_epsilon =
    ctx->network()->addElementWise(*var_output, *const_epsilon, nvinfer1::ElementWiseOperation::kSUM)->getOutput(0);
  CHECK_NULL_RETURN(var_epsilon);

  // sqrt(mean(hw*hw) - mean(hw)^2 + epsilon_), standard deviation
  auto std_dev = ctx->network()->addUnary(*var_epsilon, nvinfer1::UnaryOperation::kSQRT)->getOutput(0);
  CHECK_NULL_RETURN(std_dev);
  //  gama / sqrt(mean(hw*hw) - mean(hw)^2 + epsilon_)
  auto scale = ctx->network()->addElementWise(*gamma_, *std_dev, nvinfer1::ElementWiseOperation::kDIV)->getOutput(0);
  CHECK_NULL_RETURN(scale);

  // mean(hw)*scale
  auto mean_mul_scale =
    ctx->network()->addElementWise(*mean_output, *scale, nvinfer1::ElementWiseOperation::kPROD)->getOutput(0);
  CHECK_NULL_RETURN(mean_mul_scale);

  // bias = - mean(hw)*scale + beta_data
  auto bias =
    ctx->network()->addElementWise(*beta_, *mean_mul_scale, nvinfer1::ElementWiseOperation::kSUB)->getOutput(0);
  CHECK_NULL_RETURN(bias);

  // scale with bias: src[index]*scale
  auto scale_layer = ctx->network()->addElementWise(*reshape_output, *scale, nvinfer1::ElementWiseOperation::kPROD);
  this->layer_ = scale_layer;
  auto scale_out = scale_layer->getOutput(0);
  CHECK_NULL_RETURN(scale_out);
  // src[index]*scale - mean(hw)*scale + beta_data
  auto beta_out = ctx->network()->addElementWise(*scale_out, *bias, nvinfer1::ElementWiseOperation::kSUM)->getOutput(0);
  CHECK_NULL_RETURN(beta_out);

  auto reshape_final_layer = ctx->network()->addShuffle(*beta_out);
  reshape_final_layer->setReshapeDimensions(norm_input_dims);
  auto output = reshape_final_layer->getOutput(0);
  ctx->RegisterTensor(ITensorHelper{output, Format::NCHW, true}, out_tensors_[0].Name());
  return RET_OK;
}
REGISTER_TENSORRT_CREATOR(ops::kNameInstanceNorm, InstanceNormTensorRT)
}  // namespace mindspore::lite
