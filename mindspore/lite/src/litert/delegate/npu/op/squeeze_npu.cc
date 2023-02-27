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

#include "src/litert/delegate/npu/op/squeeze_npu.h"
namespace mindspore::lite {
int SqueezeNPUOp::Init(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                       const std::vector<mindspore::MSTensor> &out_tensors) {
  squeeze_ = new (std::nothrow) hiai::op::Squeeze(name_);
  if (squeeze_ == nullptr) {
    MS_LOG(ERROR) << "New squeeze npu operator for op " << name_ << " failed.";
    return RET_ERROR;
  }
  auto squeeze_prim = primitive->value_as_Squeeze();
  auto axis = squeeze_prim->axis();
  std::vector<int64_t> axes;
  if (axis != nullptr) {
    for (size_t i = 0; i < axis->size(); i++) {
      axes.push_back(*(axis->begin() + i));
    }
  }
  squeeze_->set_attr_axis(axes);
  return RET_OK;
}

int SqueezeNPUOp::SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                               const std::vector<mindspore::MSTensor> &out_tensors,
                               const std::vector<ge::Operator *> &npu_inputs) {
  CHECK_LESS_RETURN(npu_inputs.size(), 1);
  squeeze_->set_input_x(*npu_inputs[0]);
  return RET_OK;
}

ge::Operator *SqueezeNPUOp::GetNPUOp() { return this->squeeze_; }

SqueezeNPUOp::~SqueezeNPUOp() {
  if (squeeze_ != nullptr) {
    delete squeeze_;
    squeeze_ = nullptr;
  }
}
}  // namespace mindspore::lite
