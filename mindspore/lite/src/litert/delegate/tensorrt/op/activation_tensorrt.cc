/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "src/litert/delegate/tensorrt/op/activation_tensorrt.h"
#include <cfloat>
#include <memory>
#include <unordered_set>
#include "src/litert/delegate/tensorrt/op/cast_tensorrt.h"
#include "src/litert/delegate/tensorrt/op/activation_opt_plugin.h"

namespace mindspore::lite {
namespace {
bool HasCustomActivationPlugin(schema::ActivationType type) {
  std::unordered_set<schema::ActivationType> plugin_activation = {schema::ActivationType::ActivationType_SIGMOID,
                                                                  schema::ActivationType::ActivationType_GELU,
                                                                  schema::ActivationType::ActivationType_SWISH};
  return plugin_activation.find(type) != plugin_activation.end();
}
}  // namespace

int ActivationTensorRT::IsSupport(const schema::Primitive *primitive,
                                  const std::vector<mindspore::MSTensor> &in_tensors,
                                  const std::vector<mindspore::MSTensor> &out_tensors) {
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
  auto activation_op = this->op_primitive_->value_as_Activation();
  if (activation_op == nullptr) {
    MS_LOG(ERROR) << "op convert failed";
    return RET_ERROR;
  }
  auto activation_params_opt = TryConvertActivationType(activation_op->activation_type());
  bool has_custom_plugin = HasCustomActivationPlugin(activation_op->activation_type());
  if (!activation_params_opt && !has_custom_plugin) {
    MS_LOG(ERROR) << "Unsupported op action type for TensorRT: " << activation_op->activation_type();
    return RET_ERROR;
  }
  return RET_OK;
}
int ActivationTensorRT::AddInnerOp(TensorRTContext *ctx) {
  if (ctx->network() == nullptr) {
    MS_LOG(ERROR) << "network is invalid";
    return RET_ERROR;
  }
  auto activation_op = this->op_primitive_->value_as_Activation();
  if (activation_op == nullptr) {
    MS_LOG(ERROR) << "op convert failed";
    return RET_ERROR;
  }
  float alpha = activation_op->alpha();
  nvinfer1::ITensor *activation_input = input(ctx, 0).trt_tensor_;
  if (input(ctx, 0).trt_tensor_->getType() == nvinfer1::DataType::kINT32) {
    activation_input = TRTTensorCast(ctx, input(ctx, 0).trt_tensor_, nvinfer1::DataType::kFLOAT, op_name_ + "_cast_in");
  }

  auto runtime_precision_mode = runtime_->GetRuntimePrecisionMode();
  auto activation_layer =
    ActivationTensorRT::AddActivation(ctx, activation_op->activation_type(), alpha,
                                      std::isfinite(activation_op->min_val()) ? activation_op->min_val() : FLT_MIN,
                                      std::isfinite(activation_op->max_val()) ? activation_op->max_val() : FLT_MAX,
                                      activation_input, device_id_, quant_type_, runtime_precision_mode);
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
nvinfer1::ILayer *ActivationTensorRT::AddActivation(TensorRTContext *ctx, schema::ActivationType activation_type,
                                                    float alpha, float min_value, float max_value,
                                                    nvinfer1::ITensor *trt_in_tensor, uint32_t device_id,
                                                    schema::QuantType quant_type,
                                                    RuntimePrecisionMode runtime_precision_mode) {
  bool has_custom_plugin = HasCustomActivationPlugin(activation_type);
  // sigmoid precision is wrong for trt
  if (runtime_precision_mode == RuntimePrecisionMode::RuntimePrecisionMode_FP32 &&
      quant_type == schema::QuantType_QUANT_NONE && has_custom_plugin) {
    std::string layer_name = std::string(trt_in_tensor->getName()) + "_activation";
    auto plugin = std::make_shared<ActivationOptPlugin>(layer_name.c_str(), activation_type, device_id);
    MS_LOG(INFO) << "using opt plugin for " << layer_name;
    if (plugin == nullptr) {
      MS_LOG(ERROR) << "create ActivationOptPlugin failed for " << layer_name;
      return nullptr;
    }
    nvinfer1::ITensor *inputTensors[] = {trt_in_tensor};
    nvinfer1::IPluginV2Layer *activation_opt_layer = ctx->network()->addPluginV2(inputTensors, 1, *plugin);
    activation_opt_layer->setName(layer_name.c_str());
    return activation_opt_layer;
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

  if (activation_type == schema::ActivationType_HARD_TANH) {
    activation_layer->setAlpha(min_value);
    activation_layer->setBeta(max_value);
    return activation_layer;
  }

  if (activation_type == schema::ActivationType_SWISH) {
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
REGISTER_TENSORRT_CREATOR(schema::PrimitiveType_Activation, ActivationTensorRT)
}  // namespace mindspore::lite
