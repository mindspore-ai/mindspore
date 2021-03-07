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

#include "src/runtime/kernel/npu/reshape_npu.h"
#include <memory>
#include "src/kernel_registry.h"
#include "include/graph/op/all_ops.h"
#include "src/runtime/agent/npu/npu_converter_utils.h"
using mindspore::kernel::KERNEL_ARCH::kNPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_Reshape;

namespace mindspore::kernel {
int ReshapeNPUKernel::IsSupport(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                                OpParameter *opParameter) {
  if (reshape_param_->shape_dim_ == 0) {
    MS_LOG(ERROR) << "Npu reshape op only supports const shape.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ReshapeNPUKernel::SetNPUInputs(const std::vector<lite::Tensor *> &inputs,
                                   const std::vector<lite::Tensor *> &outputs,
                                   const std::vector<ge::Operator *> &npu_inputs) {
  op_ = new (std::nothrow) hiai::op::Reshape(name_);
  if (op_ == nullptr) {
    MS_LOG(ERROR) << name_ << " op is nullptr";
    return RET_ERROR;
  }
  op_->set_input_x(*npu_inputs[0]);
  op_->set_input_shape(*npu_inputs[1]);
  return RET_OK;
}

ge::Operator *mindspore::kernel::ReshapeNPUKernel::GetNPUOp() { return this->op_; }

ReshapeNPUKernel::~ReshapeNPUKernel() {
  if (op_ != nullptr) {
    delete op_;
    op_ = nullptr;
  }
}

REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_Reshape, NPUKernelCreator<ReshapeNPUKernel>)
}  // namespace mindspore::kernel
