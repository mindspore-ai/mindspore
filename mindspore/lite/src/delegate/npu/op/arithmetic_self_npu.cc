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

#include "src/delegate/npu/op/arithmetic_self_npu.h"
#include <string>

namespace mindspore {
template <typename T>
ge::Operator *CreateOperator(const std::string &name) {
  auto op = new (std::nothrow) T(name);
  if (op == nullptr) {
    MS_LOG(ERROR) << name << " op is nullptr";
    return nullptr;
  }
  return op;
}

int ArithmeticSelfNPUOp::Init(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                              const std::vector<mindspore::MSTensor> &out_tensors) {
  switch (type_) {
    case schema::PrimitiveType_Cos:
      op_ = CreateOperator<hiai::op::Cos>(name_);
      break;
    case schema::PrimitiveType_Log:
      op_ = CreateOperator<hiai::op::Log>(name_);
      break;
    case schema::PrimitiveType_Square:
      op_ = CreateOperator<hiai::op::Square>(name_);
      break;
    case schema::PrimitiveType_Sqrt:
      op_ = CreateOperator<hiai::op::Sqrt>(name_);
      break;
    case schema::PrimitiveType_Rsqrt:
      op_ = CreateOperator<hiai::op::Rsqrt>(name_);
      break;
    case schema::PrimitiveType_Sin:
      op_ = CreateOperator<hiai::op::Sin>(name_);
      break;
    case schema::PrimitiveType_LogicalNot:
      op_ = CreateOperator<hiai::op::LogicalNot>(name_);
      break;
    case schema::PrimitiveType_Floor:
      op_ = CreateOperator<hiai::op::Floor>(name_);
      break;
    case schema::PrimitiveType_Ceil:
      op_ = CreateOperator<hiai::op::Ceil>(name_);
      break;
    case schema::PrimitiveType_Round:
      op_ = CreateOperator<hiai::op::Round>(name_);
      break;
    case schema::PrimitiveType_Neg:
      op_ = CreateOperator<hiai::op::Neg>(name_);
      break;
    case schema::PrimitiveType_Reciprocal:
      op_ = CreateOperator<hiai::op::Reciprocal>(name_);
      break;
    default:
      MS_LOG(ERROR) << "Unsupported primitive type: " << schema::EnumNamePrimitiveType(type_);
      return RET_ERROR;
  }
  if (op_ == nullptr) {
    MS_LOG(ERROR) << "Arithmetic self create operator return nullptr.";
    return RET_ERROR;
  }
  return RET_OK;
}

template <typename T>
void SetInputs(const std::vector<ge::Operator *> &npu_inputs, ge::Operator *op) {
  auto cur_op = reinterpret_cast<T *>(op);
  cur_op->set_input_x(*npu_inputs[0]);
  return;
}

int ArithmeticSelfNPUOp::SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                                      const std::vector<mindspore::MSTensor> &out_tensors,
                                      const std::vector<ge::Operator *> &npu_inputs) {
  switch (type_) {
    case schema::PrimitiveType_Cos:
      SetInputs<hiai::op::Cos>(npu_inputs, op_);
      break;
    case schema::PrimitiveType_Log:
      SetInputs<hiai::op::Log>(npu_inputs, op_);
      break;
    case schema::PrimitiveType_Square:
      SetInputs<hiai::op::Square>(npu_inputs, op_);
      break;
    case schema::PrimitiveType_Sqrt:
      SetInputs<hiai::op::Sqrt>(npu_inputs, op_);
      break;
    case schema::PrimitiveType_Rsqrt:
      SetInputs<hiai::op::Rsqrt>(npu_inputs, op_);
      break;
    case schema::PrimitiveType_Sin:
      SetInputs<hiai::op::Sin>(npu_inputs, op_);
      break;
    case schema::PrimitiveType_LogicalNot:
      SetInputs<hiai::op::LogicalNot>(npu_inputs, op_);
      break;
    case schema::PrimitiveType_Floor:
      SetInputs<hiai::op::Floor>(npu_inputs, op_);
      break;
    case schema::PrimitiveType_Ceil:
      SetInputs<hiai::op::Ceil>(npu_inputs, op_);
      break;
    case schema::PrimitiveType_Round:
      SetInputs<hiai::op::Round>(npu_inputs, op_);
      break;
    case schema::PrimitiveType_Neg:
      SetInputs<hiai::op::Neg>(npu_inputs, op_);
      break;
    case schema::PrimitiveType_Reciprocal:
      SetInputs<hiai::op::Reciprocal>(npu_inputs, op_);
      break;
    default:
      MS_LOG(ERROR) << "SetInputs for npu op " << name_ << " failed.";
      return RET_ERROR;
  }
  return RET_OK;
}

ge::Operator *ArithmeticSelfNPUOp::GetNPUOp() { return this->op_; }

ArithmeticSelfNPUOp::~ArithmeticSelfNPUOp() {
  if (op_ != nullptr) {
    delete op_;
    op_ = nullptr;
  }
}
}  // namespace mindspore
