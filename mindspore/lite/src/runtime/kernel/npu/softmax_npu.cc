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

#include "src/runtime/kernel/npu/softmax_npu.h"
#include "src/kernel_registry.h"
using mindspore::kernel::KERNEL_ARCH::kNPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_Softmax;

namespace mindspore::kernel {
int SoftmaxNPUKernel::IsSupport(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                                OpParameter *opParameter) {
  return RET_OK;
}

int SoftmaxNPUKernel::SetNPUInputs(const std::vector<lite::Tensor *> &inputs,
                                   const std::vector<lite::Tensor *> &outputs,
                                   const std::vector<ge::Operator *> &npu_inputs) {
  op_ = new (std::nothrow) hiai::op::Softmax(name_);
  if (op_ == nullptr) {
    MS_LOG(ERROR) << name_ << " op is nullptr";
    return RET_ERROR;
  }
  if (softmax_parameter_->axis_ == -1) {
    op_->set_attr_axis(inputs[0]->shape().size() - 1);
  } else {
    op_->set_attr_axis(softmax_parameter_->axis_);
  }
  op_->set_input_x(*npu_inputs[0]);
  return RET_OK;
}

ge::Operator *mindspore::kernel::SoftmaxNPUKernel::GetNPUOp() { return this->op_; }

SoftmaxNPUKernel::~SoftmaxNPUKernel() {
  if (op_ != nullptr) {
    delete op_;
    op_ = nullptr;
  }
}

REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_Softmax, NPUKernelCreator<SoftmaxNPUKernel>)
}  // namespace mindspore::kernel
