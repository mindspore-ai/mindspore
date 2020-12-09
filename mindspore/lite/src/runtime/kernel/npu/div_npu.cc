/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "src/runtime/kernel/npu/div_npu.h"
#include "include/graph/op/all_ops.h"
#include "src/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH::kNPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_Div;

namespace mindspore::kernel {
int DivNPUKernel::IsSupport(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                            OpParameter *opParameter) {
  op_ = new hiai::op::RealDiv(name_);

  if (inputs[0]->shape() != inputs[1]->shape()) {
    MS_LOG(INFO) << "ddk 500 does not support broadcast."
                 << " shape 1 is:" << inputs[0]->shape() << " shape 2 is:" << inputs[1]->shape();
    return RET_ERROR;
  }
  return RET_OK;
}
void DivNPUKernel::SetNPUInputs(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                                const std::vector<ge::Operator *> &npu_inputs) {
  op_->set_input_x1(*npu_inputs[0]);
  op_->set_input_x2(*npu_inputs[1]);
}
ge::Operator *mindspore::kernel::DivNPUKernel::GetNPUOp() { return this->op_; }
DivNPUKernel::~DivNPUKernel() {
  if (op_ != nullptr) {
    delete op_;
    op_ = nullptr;
  }
}

REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_Div, NPUKernelCreator<DivNPUKernel>)
}  // namespace mindspore::kernel
