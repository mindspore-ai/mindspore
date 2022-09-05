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

#include "src/litert/delegate/npu/op/abs_npu.h"
#include "include/graph/op/all_ops.h"
#include "src/litert/kernel_registry.h"

namespace mindspore::lite {
int AbsNPUOp::Init(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                   const std::vector<mindspore::MSTensor> &out_tensors) {
  // NPU ddk does not support Abs op in fact. Square and Sqrt are utilized to realize it.
  square_ = new (std::nothrow) hiai::op::Square(name_ + "_square");
  if (square_ == nullptr) {
    MS_LOG(ERROR) << name_ << "_square op is nullptr";
    return RET_ERROR;
  }
  sqrt_ = new (std::nothrow) hiai::op::Sqrt(name_ + "_sqrt");
  if (sqrt_ == nullptr) {
    MS_LOG(ERROR) << name_ << "_sqrt op is nullptr";
    return RET_ERROR;
  }
  return RET_OK;
}

int AbsNPUOp::SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                           const std::vector<mindspore::MSTensor> &out_tensors,
                           const std::vector<ge::Operator *> &npu_inputs) {
  CHECK_LESS_RETURN(npu_inputs.size(), 1);
  square_->set_input_x(*npu_inputs[0]);
  sqrt_->set_input_x(*square_);
  return RET_OK;
}

ge::Operator *AbsNPUOp::GetNPUOp() { return this->sqrt_; }

AbsNPUOp::~AbsNPUOp() {
  if (square_ != nullptr) {
    delete square_;
    square_ = nullptr;
  }
  if (sqrt_ != nullptr) {
    delete sqrt_;
    sqrt_ = nullptr;
  }
}
}  // namespace mindspore::lite
