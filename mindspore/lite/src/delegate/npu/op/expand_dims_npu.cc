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

#include "src/delegate/npu/op/expand_dims_npu.h"
#include "include/graph/op/all_ops.h"
#include "src/delegate/npu/npu_converter_utils.h"

namespace mindspore {
int ExpandDimsNPUOp::Init(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                          const std::vector<mindspore::MSTensor> &out_tensors) {
  expand_dims_ = new (std::nothrow) hiai::op::ExpandDims(name_);
  if (expand_dims_ == nullptr) {
    MS_LOG(ERROR) << name_ << " op is nullptr";
    return RET_ERROR;
  }
  return RET_OK;
}

int ExpandDimsNPUOp::SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                                  const std::vector<mindspore::MSTensor> &out_tensors,
                                  const std::vector<ge::Operator *> &npu_inputs) {
  expand_dims_->set_input_x(*npu_inputs[0]);
  expand_dims_->set_input_axis(*npu_inputs[1]);
  return RET_OK;
}

ge::Operator *ExpandDimsNPUOp::GetNPUOp() { return this->expand_dims_; }

ExpandDimsNPUOp::~ExpandDimsNPUOp() {
  if (expand_dims_ != nullptr) {
    delete expand_dims_;
    expand_dims_ = nullptr;
  }
}
}  // namespace mindspore
