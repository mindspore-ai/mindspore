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

#include "src/delegate/npu/op/flatten_npu.h"
#include "include/graph/op/all_ops.h"
#include "src/delegate/npu/npu_converter_utils.h"

namespace mindspore {
int FlattenNPUOp::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                            const std::vector<mindspore::MSTensor> &out_tensors) {
  if (out_tensors.at(0).Shape().size() != C2NUM) {
    MS_LOG(WARNING) << "The output tensor can only be flatten to 2 dimension.";
    return RET_NOT_SUPPORT;
  }
  return RET_OK;
}

int FlattenNPUOp::Init(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                       const std::vector<mindspore::MSTensor> &out_tensors) {
  flatten_ = new (std::nothrow) hiai::op::Flatten(name_);
  if (flatten_ == nullptr) {
    MS_LOG(ERROR) << name_ << " op is nullptr";
    return RET_ERROR;
  }
  return RET_OK;
}

int FlattenNPUOp::SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                               const std::vector<mindspore::MSTensor> &out_tensors,
                               const std::vector<ge::Operator *> &npu_inputs) {
  flatten_->set_input_x(*npu_inputs[0]);
  return RET_OK;
}

ge::Operator *FlattenNPUOp::GetNPUOp() { return this->flatten_; }

FlattenNPUOp::~FlattenNPUOp() {
  if (flatten_ != nullptr) {
    delete flatten_;
    flatten_ = nullptr;
  }
}
}  // namespace mindspore
