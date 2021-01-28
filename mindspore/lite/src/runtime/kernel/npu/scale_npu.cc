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

#include "src/runtime/kernel/npu/scale_npu.h"
#include "src/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH::kNPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_Scale;

namespace mindspore::kernel {
int ScaleNPUKernel::IsSupport(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                              OpParameter *opParameter) {
  if (scale_parameter_->axis_ < 0) {
    scale_parameter_->axis_ = scale_parameter_->axis_ + inputs[0]->shape().size();
  }
  if (scale_parameter_->axis_ != 1) {
    MS_LOG(ERROR) << "Npu scale axis attr only support 1, now is " << scale_parameter_->axis_;
    return RET_ERROR;
  }
  return RET_OK;
}

int ScaleNPUKernel::SetNPUInputs(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                                 const std::vector<ge::Operator *> &npu_inputs) {
  op_ = new (std::nothrow) hiai::op::Scale(name_);
  if (op_ == nullptr) {
    MS_LOG(ERROR) << name_ << " op is nullptr";
    return RET_ERROR;
  }
  op_->set_attr_axis(scale_parameter_->axis_);
  op_->set_input_x(*npu_inputs[0]);
  op_->set_input_scale(*npu_inputs[1]);
  if (npu_inputs[2] != nullptr) {
    op_->set_input_bias(*npu_inputs[2]);
  }
  return RET_OK;
}

ge::Operator *mindspore::kernel::ScaleNPUKernel::GetNPUOp() { return this->op_; }

ScaleNPUKernel::~ScaleNPUKernel() {
  if (op_ != nullptr) {
    delete op_;
    op_ = nullptr;
  }
}

REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_Scale, NPUKernelCreator<ScaleNPUKernel>)
}  // namespace mindspore::kernel
