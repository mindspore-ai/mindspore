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

#include "src/litert/delegate/coreml/op/activation_coreml.h"
namespace mindspore::lite {
int ActivationCoreMLOp::IsSupport() {
  act_prim_ = op_primitive_->value_as_Activation();
  if (act_prim_ == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op ." << name_;
    return RET_ERROR;
  }
  act_type_ = act_prim_->activation_type();
  if (act_type_ != schema::ActivationType_RELU && act_type_ != schema::ActivationType_RELU6 &&
      act_type_ != schema::ActivationType_SIGMOID && act_type_ != schema::ActivationType_TANH &&
      act_type_ != schema::ActivationType_HSIGMOID && act_type_ != schema::ActivationType_LEAKY_RELU &&
      act_type_ != schema::ActivationType_SWISH && act_type_ != schema::ActivationType_ELU) {
    MS_LOG(WARNING) << "Unsupported activation type for activation op " << name_ << "when running coreML.";
    return RET_NOT_SUPPORT;
  }
  return RET_OK;
}

int ActivationCoreMLOp::BuildLayer() {
  MS_ASSERT(op_ != nullptr);
  switch (act_type_) {
    case schema::ActivationType_RELU:
      op_->mutable_activation()->mutable_relu();
      break;
    case schema::ActivationType_RELU6: {
      auto clip_param = act_op_->mutable_clip();
      clip_param->set_minval(0);
      clip_param->set_maxval(kValueThreshold6);
      break;
    }
    case schema::ActivationType_TANH:
      op_->mutable_activation()->mutable_tanh();
      break;
    case schema::ActivationType_SIGMOID:
      op_->mutable_activation()->mutable_sigmoid();
      break;
    case schema::ActivationType_LEAKY_RELU:
      op_->mutable_activation()->mutable_leakyrelu()->set_alpha(act_prim_->alpha());
      break;
    default:
      MS_LOG(ERROR) << "Unsupported activation type.";
      return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::lite
