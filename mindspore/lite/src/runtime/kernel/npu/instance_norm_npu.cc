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

#include "src/runtime/kernel/npu/instance_norm_npu.h"
#include <memory>
#include "src/kernel_registry.h"
#include "src/runtime/agent/npu/npu_converter_utils.h"

using mindspore::kernel::KERNEL_ARCH::kNPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_InstanceNorm;

namespace mindspore::kernel {
int InstanceNormNPUKernel::IsSupport(const std::vector<lite::Tensor *> &inputs,
                                     const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter) {
  return RET_OK;
}

int InstanceNormNPUKernel::SetNPUInputs(const std::vector<lite::Tensor *> &inputs,
                                        const std::vector<lite::Tensor *> &outputs,
                                        const std::vector<ge::Operator *> &npu_inputs) {
  op_ = new (std::nothrow) hiai::op::InstanceNorm(name_);
  if (op_ == nullptr) {
    MS_LOG(ERROR) << "New instance norm npu operator for op " << name_ << " failed.";
    return RET_ERROR;
  }
  op_->set_input_x(*npu_inputs[0]);

  auto gamma_shape = inputs[1]->shape();
  std::shared_ptr<ge::Tensor> gamma_tensor = std::shared_ptr<ge::Tensor>(new (std::nothrow) ge::Tensor());
  if (gamma_tensor == nullptr) {
    MS_LOG(ERROR) << "new gamma_tensor failed.";
    return RET_ERROR;
  }
  ge::TensorDesc gamma_tensor_desc(lite::ConverterToNPUShape({1, gamma_shape[0], 1, 1}), ge::FORMAT_NCHW,
                                   lite::ConverterToNPUDataType(inputs[1]->data_type()));
  gamma_tensor->SetTensorDesc(gamma_tensor_desc);
  gamma_tensor->SetData(reinterpret_cast<const uint8_t *>(inputs[1]->data_c()), inputs[1]->Size());
  gamma_ = new (std::nothrow) hiai::op::Const(name_ + "_gamma");
  if (gamma_ == nullptr) {
    MS_LOG(ERROR) << "New gamma_ const failed.";
    return RET_ERROR;
  }
  gamma_->set_attr_value(gamma_tensor);
  op_->set_input_gamma(*gamma_);

  auto beta_shape = inputs[2]->shape();
  std::shared_ptr<ge::Tensor> beta_tensor = std::shared_ptr<ge::Tensor>(new (std::nothrow) ge::Tensor());
  if (beta_tensor == nullptr) {
    MS_LOG(ERROR) << "new beta_tensor failed.";
    return RET_ERROR;
  }
  ge::TensorDesc beta_tensor_desc(lite::ConverterToNPUShape({1, beta_shape[0], 1, 1}), ge::FORMAT_NCHW,
                                  lite::ConverterToNPUDataType(inputs[2]->data_type()));
  beta_tensor->SetTensorDesc(beta_tensor_desc);
  beta_tensor->SetData(reinterpret_cast<const uint8_t *>(inputs[2]->data_c()), inputs[2]->Size());
  beta_ = new (std::nothrow) hiai::op::Const(name_ + "_beta");
  if (beta_ == nullptr) {
    MS_LOG(ERROR) << "New beta_ const failed.";
    return RET_ERROR;
  }
  beta_->set_attr_value(beta_tensor);
  op_->set_input_beta(*beta_);
  op_->set_attr_epsilon(instance_norm_param_->epsilon_);
  return RET_OK;
}

ge::Operator *mindspore::kernel::InstanceNormNPUKernel::GetNPUOp() { return this->op_; }

InstanceNormNPUKernel::~InstanceNormNPUKernel() {
  if (op_ != nullptr) {
    delete op_;
    op_ = nullptr;
  }
  if (gamma_ != nullptr) {
    delete gamma_;
    gamma_ = nullptr;
  }
  if (beta_ != nullptr) {
    delete beta_;
    beta_ = nullptr;
  }
}

REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_InstanceNorm, NPUKernelCreator<InstanceNormNPUKernel>)
}  // namespace mindspore::kernel
