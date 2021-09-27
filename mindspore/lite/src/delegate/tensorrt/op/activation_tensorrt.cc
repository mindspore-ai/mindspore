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

  nvinfer1::IActivationLayer *activation_layer = ActivationTensorRT::AddActivation(
    network, activation_op->activation_type(), alpha, tensorrt_in_tensors_[0].trt_tensor_);
  if (activation_layer == nullptr) {
    MS_LOG(ERROR) << "add activation op failed for TensorRT.";
    return RET_ERROR;
  }

  activation_layer->setName(op_name_.c_str());
  activation_layer->getOutput(0)->setName((op_name_ + "_output").c_str());
  this->AddInnerOutTensors(ITensorHelper{activation_layer->getOutput(0), tensorrt_in_tensors_[0].format_});

  return RET_OK;
}
nvinfer1::IActivationLayer *ActivationTensorRT::AddActivation(nvinfer1::INetworkDefinition *network,
                                                              schema::ActivationType activation_type, float alpha,
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

  if (action_param.has_alpha) {
    activation_layer->setAlpha(alpha);
  }

  if (action_param.has_beta) {
    activation_layer->setBeta(action_param.beta);
  }

  return activation_layer;
}
}  // namespace mindspore::lite
