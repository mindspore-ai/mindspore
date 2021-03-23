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

#include "src/runtime/kernel/npu/pad_npu.h"
#include <memory>
#include "src/kernel_registry.h"
#include "src/runtime/agent/npu/npu_converter_utils.h"
using mindspore::kernel::KERNEL_ARCH::kNPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_PadFusion;

namespace mindspore::kernel {
int PadNPUKernel::IsSupport(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                            OpParameter *opParameter) {
  if (param_->pad_mode_ != schema::PaddingMode_CONSTANT) {
    MS_LOG(WARNING) << "NPU only support CONSTANT padding mode";
    return RET_ERROR;
  }
  if (inputs.size() >= 2 && inputs[1]->data_c() != nullptr) {
    for (int i = 0; i < inputs[1]->ElementsNum(); i++) {
      param_->paddings_[i] = static_cast<int *>(inputs[1]->data_c())[i];
    }
  } else {
    MS_LOG(WARNING) << "NPU axis is attribute.";
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
  int size = static_cast<int>(param_->padding_length / 2);
  ge::TensorDesc padding_tensor_desc(ge::Shape({size, 2}), ge::FORMAT_NCHW, ge::DT_INT32);
  ge::TensorPtr padding_tensor = std::make_shared<hiai::Tensor>(padding_tensor_desc);
  padding_tensor->SetData(reinterpret_cast<uint8_t *>(param_->paddings_), 2 * size * sizeof(int));
  hiai_paddings_ = new hiai::op::Const(name_ + "paddings");
  hiai_paddings_->set_attr_value(padding_tensor);

  ge::TensorDesc constant_values_tensor_desc(ge::Shape({1}), ge::FORMAT_NCHW, ge::DT_FLOAT);
  ge::TensorPtr constant_values_tensor = std::make_shared<hiai::Tensor>(constant_values_tensor_desc);
  vector<float> constant_values_data_value = {param_->constant_value_};
  constant_values_tensor->SetData(reinterpret_cast<uint8_t *>(constant_values_data_value.data()), 1 * sizeof(float));
  hiai_constant_ = new hiai::op::Const(name_ + "constant");
  hiai_constant_->set_attr_value(constant_values_tensor);

  op_->set_input_x(*npu_inputs[0]);
  op_->set_input_constant_values(*hiai_constant_);
  op_->set_input_paddings(*hiai_paddings_);

  return RET_OK;
}

ge::Operator *mindspore::kernel::PadNPUKernel::GetNPUOp() { return this->op_; }

PadNPUKernel::~PadNPUKernel() {
  if (op_ != nullptr) {
    delete op_;
    op_ = nullptr;
  }
  if (hiai_paddings_ != nullptr) {
    delete hiai_paddings_;
    hiai_paddings_ = nullptr;
  }
  if (hiai_constant_ != nullptr) {
    delete hiai_constant_;
    hiai_constant_ = nullptr;
  }
}

REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_PadFusion, NPUKernelCreator<PadNPUKernel>)
}  // namespace mindspore::kernel
