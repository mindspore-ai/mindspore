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

#include "src/delegate/tensorrt/op/activation_tensorrt.h"
#include <cfloat>
#include <memory>
#include "src/delegate/tensorrt/op/cast_tensorrt.h"

namespace mindspore::lite {
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
  this->action_code_ = ConvertActivationType(activation_op->activation_type()).activation_type;
  if (this->action_code_ == nvinfer1::ActivationType::kRELU &&
      activation_op->activation_type() != schema::ActivationType_RELU) {
    MS_LOG(ERROR) << "Unsupported op action type for TensorRT: " << activation_op->activation_type();
    return RET_ERROR;
  }
  return RET_OK;
}
int ActivationTensorRT::AddInnerOp(nvinfer1::INetworkDefinition *network) {
  if (network == nullptr) {
    MS_LOG(ERROR) << "network is invalid";
    return RET_ERROR;
  }
  auto activation_op = this->op_primitive_->value_as_Activation();
  if (activation_op == nullptr) {
    MS_LOG(ERROR) << "op convert failed";
    return RET_ERROR;
  }
  float alpha = activation_op->alpha();
  nvinfer1::ITensor *activation_input = tensorrt_in_tensors_[0].trt_tensor_;
  if (tensorrt_in_tensors_[0].trt_tensor_->getType() == nvinfer1::DataType::kINT32) {
    auto plugin =
      std::make_shared<CastPlugin>(op_name_ + "_cast_in", nvinfer1::DataType::kINT32, nvinfer1::DataType::kFLOAT);
    nvinfer1::ITensor *inputTensors[] = {activation_input};
    nvinfer1::IPluginV2Layer *cast_layer = network->addPluginV2(inputTensors, 1, *plugin);
    if (cast_layer == nullptr) {
      MS_LOG(ERROR) << "create cast layer failed for: " << op_name_;
      return RET_ERROR;
    }
    cast_layer->setName((op_name_ + "_cast_in").c_str());
    activation_input = cast_layer->getOutput(0);
  }

  nvinfer1::IActivationLayer *activation_layer = ActivationTensorRT::AddActivation(
    network, activation_op->activation_type(), alpha,
    std::isfinite(activation_op->min_val()) ? activation_op->min_val() : FLT_MIN,
    std::isfinite(activation_op->max_val()) ? activation_op->max_val() : FLT_MAX, activation_input);
  if (activation_layer == nullptr) {
    MS_LOG(ERROR) << "add activation op failed for TensorRT.";
    return RET_ERROR;
  }

  activation_layer->setName(op_name_.c_str());
  // cast to origin type
  nvinfer1::ITensor *out_tensor = activation_layer->getOutput(0);
  if (out_tensor->getType() != ConvertDataType(out_tensors_[0].DataType())) {
    auto plugin = std::make_shared<CastPlugin>(op_name_ + "_cast_out", out_tensor->getType(),
                                               ConvertDataType(out_tensors_[0].DataType()));
    nvinfer1::ITensor *inputTensors[] = {activation_layer->getOutput(0)};
    nvinfer1::IPluginV2Layer *cast_layer = network->addPluginV2(inputTensors, 1, *plugin);
    if (cast_layer == nullptr) {
      MS_LOG(ERROR) << "create cast layer failed for: " << op_name_;
      return RET_ERROR;
    }
    cast_layer->setName((op_name_ + "_cast_out").c_str());
    out_tensor = cast_layer->getOutput(0);
  }
  out_tensor->setName((op_name_ + "_output").c_str());
  this->AddInnerOutTensors(
    ITensorHelper{out_tensor, tensorrt_in_tensors_[0].format_, tensorrt_in_tensors_[0].same_format_});
  return RET_OK;
}
nvinfer1::IActivationLayer *ActivationTensorRT::AddActivation(nvinfer1::INetworkDefinition *network,
                                                              schema::ActivationType activation_type, float alpha,
                                                              float min_value, float max_value,
                                                              nvinfer1::ITensor *trt_in_tensor) {
  // Just some action_code correct, unfind code is set to default relu. need double check.
  lite::ActivationParams action_param = ConvertActivationType(activation_type);
  if (action_param.activation_type == nvinfer1::ActivationType::kRELU &&
      activation_type != schema::ActivationType_RELU) {
    MS_LOG(ERROR) << "Unsupported op action type for TensorRT: " << activation_type;
    return nullptr;
  }
  nvinfer1::IActivationLayer *activation_layer = network->addActivation(*trt_in_tensor, action_param.activation_type);
  if (activation_layer == nullptr) {
    MS_LOG(ERROR) << "add activation op failed for TensorRT.";
    return nullptr;
  }

  if (activation_type == schema::ActivationType_HARD_TANH) {
    activation_layer->setAlpha(min_value);
    activation_layer->setBeta(max_value);
    return activation_layer;
  }

  if (action_param.has_alpha) {
    activation_layer->setAlpha(alpha);
  }

  if (action_param.has_beta) {
    activation_layer->setBeta(action_param.beta);
  }

  return activation_layer;
}
}  // namespace mindspore::lite
