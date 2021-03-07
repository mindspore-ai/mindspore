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

#include "src/runtime/kernel/npu/reduce_npu.h"
#include <memory>
#include "src/kernel_registry.h"
using mindspore::kernel::KERNEL_ARCH::kNPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_ReduceFusion;
using mindspore::schema::ReduceMode_ReduceMean;

namespace mindspore::kernel {
int ReduceNPUKernel::IsSupport(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                               OpParameter *opParameter) {
  if (reduce_param_->mode_ != ReduceMode_ReduceMean) {
    MS_LOG(ERROR) << "Npu does not support reduce mode " << reduce_param_->mode_ << " for op " << name_;
    return RET_ERROR;
  }
  if (reduce_param_->reduce_to_end_) {
    return RET_ERROR;
  }
  return RET_OK;
}

int ReduceNPUKernel::SetNPUInputs(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                                  const std::vector<ge::Operator *> &npu_inputs) {
  auto reduce_mean_ = new (std::nothrow) hiai::op::ReduceMean(name_);
  if (reduce_mean_ == nullptr) {
    MS_LOG(ERROR) << "New reduce operator for op " << name_ << " failed.";
    return RET_ERROR;
  }
  reduce_mean_->set_input_x(*npu_inputs[0])
    .set_input_axes(*npu_inputs[1])
    .set_attr_keep_dims(reduce_param_->keep_dims_);
  reduce_ = reduce_mean_;
  return RET_OK;
}

ge::Operator *mindspore::kernel::ReduceNPUKernel::GetNPUOp() { return this->reduce_; }

ReduceNPUKernel::~ReduceNPUKernel() {
  if (reduce_ != nullptr) {
    delete reduce_;
    reduce_ = nullptr;
  }
}

REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_ReduceFusion, NPUKernelCreator<ReduceNPUKernel>)
}  // namespace mindspore::kernel
