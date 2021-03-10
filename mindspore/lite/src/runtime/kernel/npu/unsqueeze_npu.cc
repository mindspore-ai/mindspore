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

#include "src/runtime/kernel/npu/unsqueeze_npu.h"
#include <memory>
#include "src/kernel_registry.h"
using mindspore::kernel::KERNEL_ARCH::kNPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_Unsqueeze;

namespace mindspore::kernel {
int UnsqueezeNPUKernel::IsSupport(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                                  OpParameter *opParameter) {
  if (inputs[0]->shape().size() > 3) {
    MS_LOG(WARNING) << "The dimension of output not support bigger than 4.";
    return RET_ERROR;
  }
  return RET_OK;
}

int UnsqueezeNPUKernel::SetNPUInputs(const std::vector<lite::Tensor *> &inputs,
                                     const std::vector<lite::Tensor *> &outputs,
                                     const std::vector<ge::Operator *> &npu_inputs) {
  op_ = new (std::nothrow) hiai::op::ExpandDims(name_);
  if (op_ == nullptr) {
    MS_LOG(ERROR) << name_ << " op is nullptr";
    return RET_ERROR;
  }
  int size = axis_.size();
  ge::TensorDesc desc(ge::Shape({size}), ge::FORMAT_NCHW, ge::DT_INT32);
  ge::TensorPtr tensor = std::make_shared<hiai::Tensor>(desc);
  tensor->SetData(reinterpret_cast<uint8_t *>(axis_.data()), size * sizeof(int));
  axis_const_ = new hiai::op::Const(name_ + "_axis");
  axis_const_->set_attr_value(tensor);

  op_->set_input_x(*npu_inputs[0]);
  op_->set_input_axis(*axis_const_);

  return RET_OK;
}

ge::Operator *mindspore::kernel::UnsqueezeNPUKernel::GetNPUOp() { return this->op_; }

UnsqueezeNPUKernel::~UnsqueezeNPUKernel() {
  if (op_ != nullptr) {
    delete op_;
    op_ = nullptr;
  }
  if (axis_const_ != nullptr) {
    delete axis_const_;
    axis_const_ = nullptr;
  }
}

REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_Unsqueeze, NPUKernelCreator<UnsqueezeNPUKernel>)
}  // namespace mindspore::kernel
