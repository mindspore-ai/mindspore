/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "src/extendrt/delegate/tensorrt/op/activation_tensorrt.h"
#include <cfloat>
#include <memory>
#include <unordered_set>
#include "src/extendrt/delegate/tensorrt/op/cast_tensorrt.h"
#include "src/extendrt/delegate/tensorrt/op/activation_opt_plugin.h"
#include "ops/fusion/activation.h"

namespace mindspore::lite {
namespace {
bool HasCustomActivationPlugin(ActivationType type) {
  std::unordered_set<ActivationType> plugin_activation = {ActivationType::SIGMOID, ActivationType::GELU,
                                                          ActivationType::SWISH};
  return plugin_activation.find(type) != plugin_activation.end();
}
}  // namespace

int ActivationTensorRT::IsSupport(const BaseOperatorPtr &base_operator, const std::vector<TensorInfo> &in_tensors,
                                  const std::vector<TensorInfo> &out_tensors) {
  if (!IsShapeKnown()) {
    MS_LOG(ERROR) << "Unsupported input tensor unknown shape: " << op_name_;
    return RET_ERROR;
  }
  if (in_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
    return RET_ERROR;
  }
  auto activation_op = AsOps<ops::Activation>();
  if (activation_op == nullptr) {
    MS_LOG(ERROR) << "op convert failed";
    return RET_ERROR;
  }
  ActivationType activation_type = ActivationType::NO_ACTIVATION;
  if (activation_op->HasAttr(ops::kActivationType)) {
    activation_type = activation_op->get_activation_type();
  }
  if (activation_type == ActivationType::HSWISH) {
    return RET_OK;
  }
  auto activation_params_opt = TryConvertActivationType(activation_type);
  if (!activation_params_opt) {
    MS_LOG(ERROR) << "Unsupported op action type for TensorRT: " << activation_type;
    return RET_ERROR;
  }
  return RET_OK;
}
int ActivationTensorRT::AddInnerOp(TensorRTContext *ctx) {
  if (ctx->network() == nullptr) {
    MS_LOG(ERROR) << "network is invalid";
    return RET_ERROR;
  }
  auto activation_op = AsOps<ops::Activation>();
  if (activation_op == nullptr) {
    MS_LOG(ERROR) << "op convert failed";
    return RET_ERROR;
  }
  float alpha = activation_op->get_alpha();
  nvinfer1::ITensor *activation_input = input(ctx, 0).trt_tensor_;
  if (input(ctx, 0).trt_tensor_->getType() == nvinfer1::DataType::kINT32) {
    activation_input = TRTTensorCast(ctx, input(ctx, 0).trt_tensor_, nvinfer1::DataType::kFLOAT, op_name_ + "_cast_in");
  }

  auto runtime_precision_mode = runtime_->GetRuntimePrecisionMode();
  ActivationType activation_type = ActivationType::NO_ACTIVATION;
  if (activation_op->HasAttr(ops::kActivationType)) {
    activation_type = activation_op->get_activation_type();
  }
  auto activation_layer = ActivationTensorRT::AddActivation(
    ctx, activation_type, alpha, std::isfinite(activation_op->get_min_val()) ? activation_op->get_min_val() : FLT_MIN,
    std::isfinite(activation_op->get_max_val()) ? activation_op->get_max_val() : FLT_MAX, activation_input, op_name_,
    device_id_, quant_type_, runtime_precision_mode);
  if (activation_layer == nullptr) {
    MS_LOG(ERROR) << "add activation op failed for TensorRT.";
    return RET_ERROR;
  }

  activation_layer->setName(op_name_.c_str());
  // cast to origin type
  nvinfer1::ITensor *out_tensor = activation_layer->getOutput(0);
  if (out_tensor->getType() != ConvertDataType(out_tensors_[0].DataType())) {
    out_tensor = TRTTensorCast(ctx, activation_layer->getOutput(0), ConvertDataType(out_tensors_[0].DataType()),
                               op_name_ + "_cast_out");
  }
  ctx->RegisterTensor(ITensorHelper{out_tensor, input(ctx, 0).format_, input(ctx, 0).same_format_},
                      out_tensors_[0].Name());
  this->layer_ = activation_layer;
  return RET_OK;
}

nvinfer1::ILayer *ActivationTensorRT::AddHSwishActivation(TensorRTContext *ctx, nvinfer1::ITensor *trt_in_tensor,
                                                          const std::string &op_name) {
  if (trt_in_tensor->getDimensions().nbDims <= 0) {
    MS_LOG(ERROR) << "Invalid input dims count " << trt_in_tensor->getDimensions().nbDims << ", " << op_name;
    return nullptr;
  }
  size_t dims_size = mindspore::IntToSize(trt_in_tensor->getDimensions().nbDims);
  static const float add_3_const = 3.0f;
  auto add_input1 =
    ConvertScalarToITensor(ctx, dims_size, &add_3_const, DataType::kNumberTypeFloat32, op_name + "_add3");
  if (add_input1 == nullptr) {
    MS_LOG(ERROR) << "Failed to add const input 3 for hard swish: " << op_name;
    return nullptr;
  }
  auto add_3 = ctx->network()->addElementWise(*trt_in_tensor, *add_input1, nvinfer1::ElementWiseOperation::kSUM);
  if (add_3 == nullptr) {
    MS_LOG(ERROR) << "Failed to add layer x+3 for hard swish: " << op_name;
    return nullptr;
  }
  add_3->setName((op_name + "_add3").c_str());
  auto add_3_output = add_3->getOutput(0);
  if (add_3_output == nullptr) {
    MS_LOG(ERROR) << "Failed to get output of layer x+3 for hard swish: " << op_name;
    return nullptr;
  }
  auto relu6 = ctx->network()->addActivation(*add_3_output, nvinfer1::ActivationType::kCLIP);
  if (relu6 == nullptr) {
    MS_LOG(ERROR) << "Failed to add layer relu6 for hard swish: " << op_name;
    return nullptr;
  }
  relu6->setAlpha(0.0f);
  relu6->setBeta(6.0f);
  relu6->setName((op_name + "_relu6").c_str());
  auto relu6_output = relu6->getOutput(0);
  if (relu6_output == nullptr) {
    MS_LOG(ERROR) << "Failed to get output of layer relu6 for hard swish: " << op_name;
    return nullptr;
  }
  auto mul = ctx->network()->addElementWise(*trt_in_tensor, *relu6_output, nvinfer1::ElementWiseOperation::kPROD);
  if (mul == nullptr) {
    MS_LOG(ERROR) << "Failed to add layer mul for hard swish: " << op_name;
    return nullptr;
  }
  mul->setName((op_name + "_mul").c_str());
  auto mul_output = mul->getOutput(0);
  if (mul_output == nullptr) {
    MS_LOG(ERROR) << "Failed to get output of layer mul for hard swish: " << op_name;
    return nullptr;
  }
  static const float div_6_const = 6.0f;
  auto div_input1 =
    ConvertScalarToITensor(ctx, dims_size, &div_6_const, DataType::kNumberTypeFloat32, op_name + "_div6");
  if (div_input1 == nullptr) {
    MS_LOG(ERROR) << "Failed to add const input 6 for hard swish: " << op_name;
    return nullptr;
  }
  auto real_div = ctx->network()->addElementWise(*mul_output, *div_input1, nvinfer1::ElementWiseOperation::kDIV);
  if (real_div == nullptr) {
    MS_LOG(ERROR) << "Failed to add layer real div for hard swish: " << op_name;
    return nullptr;
  }
  return real_div;
}

nvinfer1::ILayer *ActivationTensorRT::AddGeluActivation(TensorRTContext *ctx, nvinfer1::ITensor *trt_in_tensor,
                                                        const std::string &op_name) {
  if (trt_in_tensor->getDimensions().nbDims <= 0) {
    MS_LOG(ERROR) << "Invalid input dims count " << trt_in_tensor->getDimensions().nbDims << ", " << op_name;
    return nullptr;
  }
  auto expand_dims = [](TensorRTContext *ctx, nvinfer1::ITensor *tensor, int nbdims) {
    while (tensor->getDimensions().nbDims != nbdims) {
      tensor = ExpandDim(ctx, tensor, 0);
    }
    return tensor;
  };
  int nbdims = trt_in_tensor->getDimensions().nbDims;
  auto const_three = expand_dims(ctx, ctx->ConvertTo1DTensor(3.f), nbdims);
  auto p3 =
    ctx->network()->addElementWise(*trt_in_tensor, *const_three, nvinfer1::ElementWiseOperation::kPOW)->getOutput(0);
  auto gelu_p1 = expand_dims(ctx, ctx->ConvertTo1DTensor(0.044715f), nbdims);
  auto prod1 = ctx->network()->addElementWise(*p3, *gelu_p1, nvinfer1::ElementWiseOperation::kPROD)->getOutput(0);
  auto sum = ctx->network()->addElementWise(*prod1, *trt_in_tensor, nvinfer1::ElementWiseOperation::kSUM)->getOutput(0);
  auto gelu_p2 = expand_dims(ctx, ctx->ConvertTo1DTensor(0.7978845608f), nbdims);
  auto prod2 = ctx->network()->addElementWise(*sum, *gelu_p2, nvinfer1::ElementWiseOperation::kPROD)->getOutput(0);
  auto tanh = ctx->network()->addActivation(*prod2, nvinfer1::ActivationType::kTANH)->getOutput(0);
  auto const_one = expand_dims(ctx, ctx->ConvertTo1DTensor(1.f), nbdims);
  auto sum2 = ctx->network()->addElementWise(*const_one, *tanh, nvinfer1::ElementWiseOperation::kSUM)->getOutput(0);
  auto prod3 =
    ctx->network()->addElementWise(*sum2, *trt_in_tensor, nvinfer1::ElementWiseOperation::kPROD)->getOutput(0);
  auto gelu_p3 = expand_dims(ctx, ctx->ConvertTo1DTensor(0.5f), nbdims);
  return ctx->network()->addElementWise(*prod3, *gelu_p3, nvinfer1::ElementWiseOperation::kPROD);
}

nvinfer1::ILayer *ActivationTensorRT::AddActivation(TensorRTContext *ctx, ActivationType activation_type, float alpha,
                                                    float min_value, float max_value, nvinfer1::ITensor *trt_in_tensor,
                                                    const std::string &op_name, uint32_t device_id,
                                                    schema::QuantType quant_type,
                                                    RuntimePrecisionMode runtime_precision_mode) {
  if (activation_type == ActivationType::HSWISH) {
    return AddHSwishActivation(ctx, trt_in_tensor, op_name);
  }
  if (activation_type == ActivationType::GELU) {
    return AddGeluActivation(ctx, trt_in_tensor, op_name);
  }
  // Just some action_code correct, unfind code is set to default relu. need double check.
  auto action_param_opt = TryConvertActivationType(activation_type);
  if (!action_param_opt) {
    MS_LOG(ERROR) << "Unsupported op action type for TensorRT: " << activation_type;
    return nullptr;
  }
  auto action_param = action_param_opt.value();
  nvinfer1::IActivationLayer *activation_layer =
    ctx->network()->addActivation(*trt_in_tensor, action_param.activation_type);
  if (activation_layer == nullptr) {
    MS_LOG(ERROR) << "add activation op failed for TensorRT.";
    return nullptr;
  }

  if (activation_type == ActivationType::HARD_TANH) {
    activation_layer->setAlpha(min_value);
    activation_layer->setBeta(max_value);
    return activation_layer;
  }

  if (activation_type == ActivationType::SWISH) {
    auto sigmoid_tensor = activation_layer->getOutput(0);
    nvinfer1::ElementWiseOperation element_wise_op_ = nvinfer1::ElementWiseOperation::kPROD;
    nvinfer1::IElementWiseLayer *swish_layer =
      ctx->network()->addElementWise(*sigmoid_tensor, *trt_in_tensor, element_wise_op_);
    if (swish_layer == nullptr) {
      MS_LOG(ERROR) << "add activation op failed for TensorRT.";
      return nullptr;
    }
    return swish_layer;
  }

  if (action_param.has_alpha) {
    activation_layer->setAlpha(alpha);
  }

  if (action_param.has_beta) {
    activation_layer->setBeta(action_param.beta);
  }

  return activation_layer;
}
REGISTER_TENSORRT_CREATOR(ops::kNameActivation, ActivationTensorRT)
}  // namespace mindspore::lite
