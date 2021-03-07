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

#include "src/runtime/kernel/npu/squeeze_npu.h"
#include "src/kernel_registry.h"
using mindspore::kernel::KERNEL_ARCH::kNPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_Squeeze;

namespace mindspore::kernel {
int SqueezeNPUKernel::IsSupport(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                                OpParameter *opParameter) {
  return RET_OK;
}

int SqueezeNPUKernel::SetNPUInputs(const std::vector<lite::Tensor *> &inputs,
                                   const std::vector<lite::Tensor *> &outputs,
                                   const std::vector<ge::Operator *> &npu_inputs) {
  op_ = new (std::nothrow) hiai::op::Squeeze(name_);
  if (op_ == nullptr) {
    MS_LOG(ERROR) << "New squeeze npu operator for op " << name_ << " failed.";
    return RET_ERROR;
  }
  op_->set_input_x(*npu_inputs[0]);
  op_->set_attr_axis(axes_);
  return RET_OK;
}

ge::Operator *mindspore::kernel::SqueezeNPUKernel::GetNPUOp() { return this->op_; }

SqueezeNPUKernel::~SqueezeNPUKernel() {
  if (op_ != nullptr) {
    delete op_;
    op_ = nullptr;
  }
}

REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_Squeeze, NPUKernelCreator<SqueezeNPUKernel>)
}  // namespace mindspore::kernel
