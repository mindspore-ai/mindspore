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

#include "src/delegate/npu/op/unsqueeze_npu.h"
#include <memory>

namespace mindspore {
int UnsqueezeNPUOp::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                              const std::vector<mindspore::MSTensor> &out_tensors) {
  CHECK_LESS_RETURN(in_tensors.size(), 1);
  if (in_tensors[0].Shape().size() > INPUT_SIZE3) {
    MS_LOG(WARNING) << "The dimension of output not support bigger than 4.";
    return RET_NOT_SUPPORT;
  }
  return RET_OK;
}

int UnsqueezeNPUOp::Init(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                         const std::vector<mindspore::MSTensor> &out_tensors) {
  unsqueeze_ = new (std::nothrow) hiai::op::ExpandDims(name_);
  if (unsqueeze_ == nullptr) {
    MS_LOG(ERROR) << "New unsqueeze npu operator for op " << name_ << " failed.";
    return RET_ERROR;
  }

  auto unsqueeze_prim = primitive->value_as_Unsqueeze();
  if (unsqueeze_prim == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op ." << name_;
    return RET_ERROR;
  }
  axis_ = std::vector<int>(unsqueeze_prim->axis()->begin(), unsqueeze_prim->axis()->end());
  int size = axis_.size();
  ge::TensorDesc desc(ge::Shape({size}), ge::FORMAT_NCHW, ge::DT_INT32);
  ge::TensorPtr tensor = std::make_shared<hiai::Tensor>(desc);
  tensor->SetData(reinterpret_cast<uint8_t *>(axis_.data()), size * sizeof(int));
  axis_const_ = new hiai::op::Const(name_ + "_axis");
  if (axis_const_ == nullptr) {
    MS_LOG(ERROR) << "create const NPU op failed for " << name_;
    return RET_ERROR;
  }
  axis_const_->set_attr_value(tensor);
  unsqueeze_->set_input_axis(*axis_const_);
  return RET_OK;
}

int UnsqueezeNPUOp::SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                                 const std::vector<mindspore::MSTensor> &out_tensors,
                                 const std::vector<ge::Operator *> &npu_inputs) {
  CHECK_NULL_RETURN(unsqueeze_);
  CHECK_LESS_RETURN(npu_inputs.size(), 1);
  unsqueeze_->set_input_x(*npu_inputs[0]);
  return RET_OK;
}

ge::Operator *UnsqueezeNPUOp::GetNPUOp() { return this->unsqueeze_; }

UnsqueezeNPUOp::~UnsqueezeNPUOp() {
  if (unsqueeze_ != nullptr) {
    delete unsqueeze_;
    unsqueeze_ = nullptr;
  }
  if (axis_const_ != nullptr) {
    delete axis_const_;
    axis_const_ = nullptr;
  }
}
}  // namespace mindspore
