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
#include "src/delegate/tensorrt/tensorrt_runtime.h"

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

void TensorRTOp::SetRuntime(TensorRTRuntime *runtime) { this->runtime_ = runtime; }

bool TensorRTOp::IsShapeKnown() {
  if (this->in_tensors_.size() == 1 && this->in_tensors_[0].Shape().size() == 0) {
    return false;
  }
  return true;
}
int TensorRTOp::Prepare(void **network_tensor_bindings, nvinfer1::ICudaEngine *engine) {
  if (op_binding_tensor_.size() != 0) {
    MS_LOG(ERROR) << "need special op Prepare for " << op_name_;
    return RET_ERROR;
  }
  return RET_OK;
}
DynamicShapeParams TensorRTOp::GetDynamicShapeParams() const { return this->dynamic_shape_params_; }
}  // namespace mindspore::lite
