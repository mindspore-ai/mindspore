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

#include "src/runtime/kernel/npu/pad_npu.h"
#include <memory>
#include "src/kernel_registry.h"
#include "src/runtime/agent/npu/npu_converter_utils.h"
using mindspore::kernel::KERNEL_ARCH::kNPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_Pad;

namespace mindspore::kernel {
int PadNPUKernel::IsSupport(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                            OpParameter *opParameter) {
  if (pad_->GetPaddingMode() != schema::PaddingMode_CONSTANT) {
    MS_LOG(WARNING) << "NPU only support CONSTANT padding mode";
    return RET_ERROR;
  }
  return RET_OK;
}

int PadNPUKernel::SetNPUInputs(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                               const std::vector<ge::Operator *> &npu_inputs) {
  op_ = new (std::nothrow) hiai::op::PadV2(name_);
  if (op_ == nullptr) {
    MS_LOG(ERROR) << name_ << " op is nullptr";
    return RET_ERROR;
  }
  int size = static_cast<int>(pad_->GetPaddings().size() / 2);
  ge::TensorDesc padding_tensor_desc(ge::Shape({size, 2}), ge::FORMAT_NCHW, ge::DT_INT32);
  ge::TensorPtr padding_tensor = std::make_shared<hiai::Tensor>(padding_tensor_desc);
  padding_tensor->SetData(reinterpret_cast<uint8_t *>(pad_->GetPaddings().data()), 2 * size * sizeof(int));
  auto paddings = new hiai::op::Const(name_ + "paddings");
  paddings->set_attr_value(padding_tensor);

  ge::TensorDesc constant_values_tensor_desc(ge::Shape({1}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  ge::TensorPtr constant_values_tensor = std::make_shared<hiai::Tensor>(constant_values_tensor_desc);
  vector<float> constant_values_data_value = {pad_->GetConstantValue()};
  constant_values_tensor->SetData(reinterpret_cast<uint8_t *>(constant_values_data_value.data()), 1 * sizeof(float));
  auto constant = new hiai::op::Const(name_ + "constant");
  constant->set_attr_value(constant_values_tensor);

  op_->set_input_x(*npu_inputs[0]);
  op_->set_input_constant_values(*constant);
  op_->set_input_paddings(*paddings);

  return RET_OK;
}

ge::Operator *mindspore::kernel::PadNPUKernel::GetNPUOp() { return this->op_; }

PadNPUKernel::~PadNPUKernel() {
  if (op_ != nullptr) {
    delete op_;
    op_ = nullptr;
  }
}

REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_Pad, NPUKernelCreator<PadNPUKernel>)
}  // namespace mindspore::kernel
