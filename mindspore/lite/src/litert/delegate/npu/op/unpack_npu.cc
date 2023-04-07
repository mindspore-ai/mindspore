/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include "src/litert/delegate/npu/op/unpack_npu.h"
#include "src/litert/delegate/npu/npu_converter_utils.h"

namespace mindspore {
namespace lite {
int UnpackNPUOp::Init(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                      const std::vector<mindspore::MSTensor> &out_tensors) {
  unpack_ = new (std::nothrow) hiai::op::Unpack(name_);
  if (unpack_ == nullptr) {
    MS_LOG(ERROR) << "New unpack npu operator for op " << name_ << " failed.";
    return RET_ERROR;
  }
  auto unstack_prim = primitive->value_as_Unstack();
  if (unstack_prim == nullptr) {
    MS_LOG(ERROR) << "Get null primitive value for op " << name_;
    return RET_ERROR;
  }

  CHECK_LESS_RETURN(in_tensors.size(), 1);
  auto in_tensor = in_tensors.at(0);
  auto axis = static_cast<int>(unstack_prim->axis());
  axis_ = axis >= 0 ? axis : axis + static_cast<int>(in_tensor.Shape().size());
  MS_CHECK_TRUE_MSG(axis_ >= 0, RET_ERROR, "The unstack axis is illegal!");
  auto num = in_tensor.Shape().at(axis_);
  unpack_->set_attr_num(num);
  unpack_->create_dynamic_output_y(num);
  return RET_OK;
}

int UnpackNPUOp::SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                              const std::vector<mindspore::MSTensor> &out_tensor,
                              const std::vector<ge::Operator *> &npu_inputs) {
  unpack_->set_input_x(*npu_inputs[0]);
  unpack_->set_attr_axis(axis_);
  return RET_OK;
}

ge::Operator *UnpackNPUOp::GetNPUOp() { return this->unpack_; }

int UnpackNPUOp::HandleAxisAndConstantInputs(std::vector<mindspore::MSTensor *> *all_tensors) {
  axis_ = TransFormAxis(axis_);
  if (axis_ == NCHW_INVALID) {
    MS_LOG(ERROR) << "Transform axis for unstack op failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

UnpackNPUOp::~UnpackNPUOp() {
  if (unpack_ != nullptr) {
    delete unpack_;
    unpack_ = nullptr;
  }
}
}  // namespace lite
}  // namespace mindspore
