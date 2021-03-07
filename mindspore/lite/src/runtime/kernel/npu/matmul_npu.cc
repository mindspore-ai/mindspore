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

#include "src/runtime/kernel/npu/matmul_npu.h"
#include "src/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH::kNPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_MatMul;

namespace mindspore::kernel {
int MatMulNPUKernel::IsSupport(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                               OpParameter *opParameter) {
  return RET_OK;
}

int MatMulNPUKernel::SetNPUInputs(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                                  const std::vector<ge::Operator *> &npu_inputs) {
  op_ = new (std::nothrow) hiai::op::MatMul(name_);
  op_->set_input_x1(*npu_inputs[0]);
  op_->set_input_x2(*npu_inputs[1]);
  if (npu_inputs.size() == 3) {
    op_->set_input_bias(*npu_inputs[2]);
  }

  op_->set_attr_transpose_x1(matmul_parameter_->a_transpose_);
  op_->set_attr_transpose_x2(matmul_parameter_->b_transpose_);
  return RET_OK;
}

ge::Operator *mindspore::kernel::MatMulNPUKernel::GetNPUOp() { return this->op_; }

MatMulNPUKernel::~MatMulNPUKernel() {
  if (op_ != nullptr) {
    delete op_;
    op_ = nullptr;
  }
}

REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_MatMul, NPUKernelCreator<MatMulNPUKernel>)
}  // namespace mindspore::kernel
