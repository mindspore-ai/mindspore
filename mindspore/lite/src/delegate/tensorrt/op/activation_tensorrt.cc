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
#include "src/delegate/tensorrt/tensorrt_utils.h"

namespace mindspore::lite {
int ActivationTensorRT::IsSupport(const schema::Primitive *primitive, const std::vector<tensor::MSTensor *> &in_tensors,
                                  const std::vector<tensor::MSTensor *> &out_tensors) {
  if (in_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported input tensor size, size is " << in_tensors.size();
    return RET_ERROR;
  }
  if (out_tensors.size() != 1) {
    MS_LOG(ERROR) << "Unsupported output tensor size, size is " << out_tensors.size();
    return RET_ERROR;
  }
  if (type_ != schema::PrimitiveType_Activation) {
    MS_LOG(ERROR) << "Unsupported schema type:" << schema::EnumNamePrimitiveType(type_);
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
  nvinfer1::ActivationType action_code = ConvertActivationType(activation_op->activation_type());

  nvinfer1::IActivationLayer *activation_layer = network->addActivation(*tensorrt_in_tensors_[0], action_code);
  if (activation_layer == nullptr) {
    MS_LOG(ERROR) << "add activation op failed for TensorRT.";
    return RET_ERROR;
  }

  if (activation_op->alpha() != activation_layer->getAlpha()) {
    activation_layer->setAlpha(activation_op->alpha());
  }
  activation_layer->setName(op_name_.c_str());
  this->AddInnerOutTensors(activation_layer->getOutput(0));

  return RET_OK;
}
}  // namespace mindspore::lite
