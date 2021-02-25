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

#include "src/runtime/kernel/npu/split_npu.h"
#include <memory>
#include "src/kernel_registry.h"
#include "src/runtime/agent/npu/npu_converter_utils.h"
using mindspore::kernel::KERNEL_ARCH::kNPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_Split;

namespace mindspore::kernel {
int SplitNPUKernel::IsSupport(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                              OpParameter *opParameter) {
  return RET_OK;
}

int SplitNPUKernel::SetNPUInputs(const std::vector<lite::Tensor *> &inputs, const std::vector<lite::Tensor *> &outputs,
                                 const std::vector<ge::Operator *> &npu_inputs) {
  op_ = new (std::nothrow) hiai::op::SplitV(name_);
  if (op_ == nullptr) {
    MS_LOG(ERROR) << name_ << " op is nullptr";
    return RET_ERROR;
  }
  int size = param_->num_split_;
  ge::TensorDesc size_splits_tensor_desc(ge::Shape({size}), ge::FORMAT_NCHW, ge::DT_INT32);
  ge::TensorPtr size_splits_tensor = std::make_shared<hiai::Tensor>(size_splits_tensor_desc);
  size_splits_tensor->SetData(reinterpret_cast<uint8_t *>(param_->split_sizes_), size * sizeof(int));
  size_splits_ = new hiai::op::Const(name_ + "_size");
  size_splits_->set_attr_value(size_splits_tensor);

  ge::TensorDesc split_dim_tensor_desc(ge::Shape({1}), ge::FORMAT_NCHW, ge::DT_INT32);
  ge::TensorPtr split_dim_tensor = std::make_shared<hiai::Tensor>(split_dim_tensor_desc);
  vector<int32_t> split_dim_data_value = {param_->split_dim_};
  split_dim_tensor->SetData(reinterpret_cast<uint8_t *>(split_dim_data_value.data()), 1 * sizeof(int));
  split_dim_ = new hiai::op::Const(name_ + "_dim");
  split_dim_->set_attr_value(split_dim_tensor);

  op_->set_input_x(*npu_inputs[0]);
  op_->set_attr_num_split(param_->num_split_);
  op_->set_input_split_dim(*split_dim_);
  op_->set_input_size_splits(*size_splits_);
  op_->create_dynamic_output_y(param_->num_split_);
  return RET_OK;
}

ge::Operator *mindspore::kernel::SplitNPUKernel::GetNPUOp() { return this->op_; }

SplitNPUKernel::~SplitNPUKernel() {
  if (op_ != nullptr) {
    delete op_;
    op_ = nullptr;
  }
  if (size_splits_ != nullptr) {
    delete size_splits_;
    size_splits_ = nullptr;
  }
  if (split_dim_ != nullptr) {
    delete split_dim_;
    split_dim_ = nullptr;
  }
}

REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_Split, NPUKernelCreator<SplitNPUKernel>)
}  // namespace mindspore::kernel
