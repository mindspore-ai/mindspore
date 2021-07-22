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

#include "src/delegate/npu/op/slice_npu.h"
#include "src/delegate/npu/npu_converter_utils.h"

namespace mindspore {
constexpr int OFFSET_INDEX = 1;
constexpr int SLICE_SIZE_INDEX = 2;

int SliceNPUOp::Init(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                     const std::vector<mindspore::MSTensor> &out_tensors) {
  slice_ = new (std::nothrow) hiai::op::Slice(name_);
  if (slice_ == nullptr) {
    MS_LOG(ERROR) << name_ << " op is nullptr";
    return RET_ERROR;
  }
  return RET_OK;
}

int SliceNPUOp::SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                             const std::vector<mindspore::MSTensor> &out_tensors,
                             const std::vector<ge::Operator *> &npu_inputs) {
  slice_->set_input_x(*npu_inputs[0]);
  slice_->set_input_offsets(*npu_inputs[OFFSET_INDEX]);
  slice_->set_input_size(*npu_inputs[SLICE_SIZE_INDEX]);
  return RET_OK;
}

ge::Operator *SliceNPUOp::GetNPUOp() { return this->slice_; }

SliceNPUOp::~SliceNPUOp() {
  if (slice_ != nullptr) {
    delete slice_;
    slice_ = nullptr;
  }
}
}  // namespace mindspore
