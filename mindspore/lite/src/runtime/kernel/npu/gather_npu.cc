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

#include "src/runtime/kernel/npu/gather_npu.h"
#include "src/kernel_registry.h"
#include "src/runtime/agent/npu/npu_converter_utils.h"
using mindspore::kernel::KERNEL_ARCH::kNPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_Gather;

namespace mindspore::kernel {
int GatherNPUKernel::IsSupport(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                               OpParameter *opParameter) {
  if (inputs[1]->data_type() != kNumberTypeInt32) {
    MS_LOG(WARNING) << "Gather indices only support Int32";
    return RET_ERROR;
  }
  if (inputs.size() >= 3 && inputs[2]->ElementsNum() == 1) {
    axis_ = static_cast<int *>(inputs[2]->data_c())[0];
  } else {
    MS_LOG(WARNING) << "NPU axis is attribute.";
    return RET_ERROR;
  }
  return RET_OK;
}

int GatherNPUKernel::SetNPUInputs(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                                  const std::vector<ge::Operator *> &npu_inputs) {
  op_ = new (std::nothrow) hiai::op::GatherV2D(name_);
  if (op_ == nullptr) {
    MS_LOG(ERROR) << name_ << " op is nullptr";
    return RET_ERROR;
  }

  op_->set_input_x(*npu_inputs[0]);
  op_->set_input_indices(*npu_inputs[1]);
  op_->set_attr_axis(axis_);
  return RET_OK;
}

ge::Operator *mindspore::kernel::GatherNPUKernel::GetNPUOp() { return this->op_; }

GatherNPUKernel::~GatherNPUKernel() {
  if (op_ != nullptr) {
    delete op_;
    op_ = nullptr;
  }
}
// NPU input index 0 datatype not support: 3(int32).
REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_Gather, NPUKernelCreator<GatherNPUKernel>)
}  // namespace mindspore::kernel
