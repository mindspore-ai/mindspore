/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "src/runtime/delegate/npu/op/softmax_npu.h"
#include "src/runtime/delegate/npu/npu_converter_utils.h"

namespace mindspore {
int SoftmaxNPUOp::Init(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                       const std::vector<mindspore::MSTensor> &out_tensors) {
  softmax_ = new (std::nothrow) hiai::op::Softmax(name_);
  if (softmax_ == nullptr) {
    MS_LOG(ERROR) << name_ << " op is nullptr";
    return RET_ERROR;
  }
  auto softmax_prim = primitive->value_as_Softmax();
  if (softmax_prim == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op ." << name_;
    return RET_ERROR;
  }
  axis_ = static_cast<int>(*(softmax_prim->axis()->begin()));
  if (axis_ == -1) {
    axis_ = in_tensors[0].Shape().size() + axis_;
  }
  return RET_OK;
}

int SoftmaxNPUOp::SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                               const std::vector<mindspore::MSTensor> &out_tensors,
                               const std::vector<ge::Operator *> &npu_inputs) {
  softmax_->set_input_x(*npu_inputs[0]);
  softmax_->set_attr_axis(axis_);
  return RET_OK;
}

ge::Operator *SoftmaxNPUOp::GetNPUOp() { return this->softmax_; }

int SoftmaxNPUOp::HandleAxisAndConstantInputs(std::vector<mindspore::MSTensor *> *all_tensors) {
  axis_ = TransFormAxis(axis_);
  if (axis_ == NCHW_INVALID) {
    MS_LOG(ERROR) << "Transform axis for Softmax op failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

SoftmaxNPUOp::~SoftmaxNPUOp() {
  if (softmax_ != nullptr) {
    delete softmax_;
    softmax_ = nullptr;
  }
}
}  // namespace mindspore
