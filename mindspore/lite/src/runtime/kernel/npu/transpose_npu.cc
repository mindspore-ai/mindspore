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

#include "src/runtime/kernel/npu/transpose_npu.h"
#include "src/kernel_registry.h"
#include "src/runtime/agent/npu/npu_converter_utils.h"
using mindspore::kernel::KERNEL_ARCH::kNPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_Transpose;

namespace mindspore::kernel {
int TransposeNPUKernel::IsSupport(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                                  OpParameter *opParameter) {
  if (conjugate_) {
    MS_LOG(ERROR) << "Unsupported conjugate transpose.";
    return RET_ERROR;
  }
  if (inputs.size() >= 2 && inputs[1]->data_c() != nullptr) {
    for (int i = 0; i < inputs[1]->ElementsNum(); i++) {
      perm_.push_back(static_cast<int *>(inputs[1]->data_c())[i]);
    }
  } else {
    MS_LOG(WARNING) << "NPU perm is attribute.";
    return RET_ERROR;
  }

  return RET_ERROR;
}

int TransposeNPUKernel::SetNPUInputs(const std::vector<lite::Tensor *> &inputs,
                                     const std::vector<lite::Tensor *> &outputs,
                                     const std::vector<ge::Operator *> &npu_inputs) {
  op_ = new (std::nothrow) hiai::op::Permute(name_);
  if (op_ == nullptr) {
    MS_LOG(ERROR) << name_ << " op is nullptr";
    return RET_ERROR;
  }
  op_->set_input_x(*npu_inputs[0]);
  op_->set_attr_order(perm_);

  return RET_OK;
}

ge::Operator *mindspore::kernel::TransposeNPUKernel::GetNPUOp() { return this->op_; }

TransposeNPUKernel::~TransposeNPUKernel() {
  if (op_ != nullptr) {
    delete op_;
    op_ = nullptr;
  }
}

REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_Transpose, NPUKernelCreator<TransposeNPUKernel>)
}  // namespace mindspore::kernel
