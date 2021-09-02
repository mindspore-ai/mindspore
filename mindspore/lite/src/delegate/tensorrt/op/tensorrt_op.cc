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

#include "src/delegate/tensorrt/op/tensorrt_op.h"

namespace mindspore::lite {
const schema::Primitive *TensorRTOp::GetPrimitive() { return this->op_primitive_; }

void TensorRTOp::AddInnerInTensors(ITensorHelper tensor) { this->tensorrt_in_tensors_.push_back(tensor); }

void TensorRTOp::AddInnerOutTensors(ITensorHelper tensor) { this->tensorrt_out_tensors_.push_back(tensor); }

std::vector<ITensorHelper> &TensorRTOp::GetInnerOutTensor() { return this->tensorrt_out_tensors_; }

std::vector<ITensorHelper> &TensorRTOp::GetInnerInTensors() { return this->tensorrt_in_tensors_; }

std::string TensorRTOp::GetOpName() { return this->op_name_; }

std::vector<mindspore::MSTensor> &TensorRTOp::inputs() { return this->in_tensors_; }

std::vector<mindspore::MSTensor> &TensorRTOp::outputs() { return this->out_tensors_; }

schema::PrimitiveType TensorRTOp::type() const { return this->type_; }

void TensorRTOp::set_in_ops(const std::vector<TensorRTOp *> &in_ops) { this->in_ops_ = in_ops; }

void TensorRTOp::set_out_ops(const std::vector<TensorRTOp *> &out_ops) { this->out_ops_ = out_ops; }

const std::vector<TensorRTOp *> &TensorRTOp::in_ops() const { return this->in_ops_; }

const std::vector<TensorRTOp *> &TensorRTOp::out_ops() const { return this->out_ops_; }

bool TensorRTOp::IsShapeKnown() {
  if (this->in_tensors_[0].Shape().size() == 0) {
    return false;
  } else {
    if (this->in_tensors_[0].Shape()[0] == -1) {
      return false;
    }
  }
  return true;
}
}  // namespace mindspore::lite
