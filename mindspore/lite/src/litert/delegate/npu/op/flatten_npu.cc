/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "src/litert/delegate/npu/op/flatten_npu.h"
#include "include/graph/op/all_ops.h"
#include "src/litert/delegate/npu/npu_converter_utils.h"
#include "src/litert/delegate/npu/npu_manager.h"

namespace mindspore::lite {
int FlattenNPUOp::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                            const std::vector<mindspore::MSTensor> &out_tensors) {
  if (out_tensors.at(0).Shape().size() != C2NUM) {
    MS_LOG(WARNING) << "The output tensor can only be flatten to 2 dimension.";
    return RET_NOT_SUPPORT;
  }
  use_reshape_ = !NPUManager::CheckDDKVerGreatEqual("100.500.010.045");
  return RET_OK;
}

int FlattenNPUOp::Init(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                       const std::vector<mindspore::MSTensor> &out_tensors) {
  if (use_reshape_) {
    reshape_ = new (std::nothrow) hiai::op::Reshape(name_ + "_reshape");
    if (reshape_ == nullptr) {
      MS_LOG(ERROR) << "New Reshape operator for op " << name_ << " failed.";
      return RET_ERROR;
    }
  } else {
    flatten_ = new (std::nothrow) hiai::op::Flatten(name_);
    if (flatten_ == nullptr) {
      MS_LOG(ERROR) << "New Flatten operator for op " << name_ << " failed.";
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int FlattenNPUOp::SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                               const std::vector<mindspore::MSTensor> &out_tensors,
                               const std::vector<ge::Operator *> &npu_inputs) {
  CHECK_LESS_RETURN(npu_inputs.size(), 1);
  if (use_reshape_) {
    auto output_shape = out_tensors.front().Shape();
    int64_t dims = output_shape.size();
    std::vector<int> valid_shape;
    for (int i = 0; i < dims; i++) {
      valid_shape.emplace_back(static_cast<int>(output_shape.at(i)));
    }
    auto valid_data_ptr = reinterpret_cast<const uint8_t *>(valid_shape.data());
    shape_ = GetNPUConst<int>(valid_data_ptr, {dims}, ge::DT_INT32, name_ + "_shape");
    if (shape_ == nullptr) {
      MS_LOG(ERROR) << "Get NPU Const for Reshape failed.";
      return RET_ERROR;
    }
    reshape_->set_input_x(*npu_inputs[0]);
    reshape_->set_input_shape(*shape_);
  } else {
    flatten_->set_input_x(*npu_inputs[0]);
  }
  return RET_OK;
}

ge::Operator *FlattenNPUOp::GetNPUOp() {
  if (use_reshape_) {
    return this->reshape_;
  } else {
    return this->flatten_;
  }
}

FlattenNPUOp::~FlattenNPUOp() {
  if (flatten_ != nullptr) {
    delete flatten_;
    flatten_ = nullptr;
  }
  if (reshape_ != nullptr) {
    delete reshape_;
    reshape_ = nullptr;
  }
  if (shape_ != nullptr) {
    delete shape_;
    shape_ = nullptr;
  }
}
}  // namespace mindspore::lite
