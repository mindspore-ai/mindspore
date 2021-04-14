/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include "src/runtime/kernel/npu/tile_npu.h"
#include <memory>
#include "include/graph/op/all_ops.h"
#include "src/kernel_registry.h"
#include "src/runtime/agent/npu/npu_converter_utils.h"

using mindspore::kernel::KERNEL_ARCH::kNPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_TileFusion;

namespace mindspore::kernel {
int TileNPUKernel::IsSupport(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                             OpParameter *opParameter) {
  if (inputs.size() != 2) {
    return RET_ERROR;
  }
  auto multiple_tensor = inputs[1];
  if (multiple_tensor->ElementsNum() > 4) {
    return RET_ERROR;
  }
  int *multiple_data = reinterpret_cast<int *>(multiple_tensor->data_c());
  if (multiple_data == nullptr) {
    return RET_ERROR;
  }
  for (int i = 0; i < multiple_tensor->ElementsNum(); ++i) {
    param_->multiples_[i] = multiple_data[i];
  }
  param_->multiples_size_ = static_cast<size_t>(multiple_tensor->ElementsNum());
  return RET_OK;
}

int TileNPUKernel::SetNPUInputs(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                                const std::vector<ge::Operator *> &npu_inputs) {
  op_ = new (std::nothrow) hiai::op::Tile(name_);
  if (op_ == nullptr) {
    MS_LOG(ERROR) << name_ << " op is nullptr";
    return RET_ERROR;
  }
  op_->set_input_x(*npu_inputs[0]);

  ge::TensorDesc multiple_tensor_desc(ge::Shape({static_cast<int64_t>(param_->multiples_size_)}), ge::FORMAT_NCHW,
                                      ge::DT_INT32);
  ge::TensorPtr multiple_tensor = std::make_shared<hiai::Tensor>(multiple_tensor_desc);
  multiple_tensor->SetData(reinterpret_cast<uint8_t *>(param_->multiples_), param_->multiples_size_ * sizeof(int));
  multiple_ = new hiai::op::Const(name_ + "multiples");
  multiple_->set_attr_value(multiple_tensor);
  op_->set_input_multiples(*multiple_);
  return RET_OK;
}

ge::Operator *mindspore::kernel::TileNPUKernel::GetNPUOp() { return this->op_; }

TileNPUKernel::~TileNPUKernel() {
  if (op_ != nullptr) {
    delete op_;
    op_ = nullptr;
  }
  if (multiple_ != nullptr) {
    delete multiple_;
    multiple_ = nullptr;
  }
}
REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_TileFusion, NPUKernelCreator<TileNPUKernel>)
}  // namespace mindspore::kernel
