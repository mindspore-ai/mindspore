/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include "src/runtime/kernel/npu/activation_npu.h"
#include "include/graph/op/all_ops.h"
#include "src/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH::kNPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_Activation;

namespace mindspore::kernel {
int ActivationNPUKernel::IsSupport(const std::vector<lite::Tensor *> &inputs,
                                   const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter) {
  if (act_param_->type_ != schema::ActivationType_RELU && act_param_->type_ != schema::ActivationType_RELU6 &&
      act_param_->type_ != schema::ActivationType_SIGMOID && act_param_->type_ != schema::ActivationType_TANH &&
      act_param_->type_ != schema::ActivationType_HSIGMOID && act_param_->type_ != schema::ActivationType_LEAKY_RELU) {
    MS_LOG(ERROR) << "Unsupported activation type for activation op " << name_ << "when running npu";
    return RET_ERROR;
  }
  return RET_OK;
}

int ActivationNPUKernel::SetNPUInputs(const std::vector<lite::Tensor *> &inputs,
                                      const std::vector<lite::Tensor *> &outputs,
                                      const std::vector<ge::Operator *> &npu_inputs) {
  act_ = new (std::nothrow) hiai::op::Activation(name_);
  if (act_ == nullptr) {
    MS_LOG(ERROR) << "New activation npu operator for activation op " << name_ << " failed.";
    return RET_ERROR;
  }
  act_->set_input_x(*npu_inputs[0]);
  switch (act_param_->type_) {
    case schema::ActivationType_SIGMOID:
      act_->set_attr_mode(0);
      break;
    case schema::ActivationType_RELU:
      act_->set_attr_mode(1);
      break;
    case schema::ActivationType_TANH:
      act_->set_attr_mode(2);
      break;
    case schema::ActivationType_LEAKY_RELU:
      act_->set_attr_mode(5);
      act_->set_attr_negative_slope(act_param_->alpha_);
      break;
    case schema::ActivationType_HSIGMOID:
      act_->set_attr_mode(10);
      break;
    case schema::ActivationType_RELU6:
      act_->set_attr_mode(14);
      break;
    default:
      MS_LOG(ERROR) << "Unsupported activation type for activation op " << name_ << "when running npu";
      return RET_ERROR;
  }
  return RET_OK;
}

ge::Operator *mindspore::kernel::ActivationNPUKernel::GetNPUOp() { return act_; }

ActivationNPUKernel::~ActivationNPUKernel() {
  if (act_ != nullptr) {
    delete act_;
    act_ = nullptr;
  }
}

REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_Activation, NPUKernelCreator<ActivationNPUKernel>)
}  // namespace mindspore::kernel
