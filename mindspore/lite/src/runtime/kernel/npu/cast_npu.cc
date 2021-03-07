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

#include "src/runtime/kernel/npu/cast_npu.h"
#include "src/kernel_registry.h"
#include "src/runtime/agent/npu/npu_converter_utils.h"
using mindspore::kernel::KERNEL_ARCH::kNPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_Cast;

namespace mindspore::kernel {
int CastNPUKernel::IsSupport(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                             OpParameter *opParameter) {
  if (inputs.size() >= 2 && inputs[1]->ElementsNum() == 1) {
    dst_type_ = static_cast<int *>(inputs[1]->data_c())[0];
  } else {
    MS_LOG(WARNING) << "NPU dst dtype is attribute.";
    return RET_ERROR;
  }
  return RET_OK;
}

int CastNPUKernel::SetNPUInputs(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                                const std::vector<ge::Operator *> &npu_inputs) {
  op_ = new (std::nothrow) hiai::op::CastT(name_);
  if (op_ == nullptr) {
    MS_LOG(ERROR) << name_ << " op is nullptr";
    return RET_ERROR;
  }
  op_->set_input_x(*npu_inputs[0]);
  op_->set_attr_dst_dtype(lite::ConverterToNPUDataType(static_cast<TypeId>(dst_type_)));
  op_->set_attr_src_dtype(lite::ConverterToNPUDataType(static_cast<TypeId>(inputs[0]->data_type())));
  return RET_OK;
}

ge::Operator *mindspore::kernel::CastNPUKernel::GetNPUOp() { return this->op_; }

CastNPUKernel::~CastNPUKernel() {
  if (op_ != nullptr) {
    delete op_;
    op_ = nullptr;
  }
}

REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_Cast, NPUKernelCreator<CastNPUKernel>)
REG_KERNEL(kNPU, kNumberTypeInt32, PrimitiveType_Cast, NPUKernelCreator<CastNPUKernel>)
}  // namespace mindspore::kernel
