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

#include "src/delegate/npu/op/arithmetic_npu.h"
#include "src/delegate/npu/npu_converter_utils.h"
namespace mindspore {
constexpr int ARITHMETIC_INPUT_NUM = 2;
int ArithmeticNPUOp::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                               const std::vector<mindspore::MSTensor> &out_tensors) {
  if (in_tensors[0].Shape().size() != 0 && in_tensors[1].Shape().size() != 0 &&
      in_tensors[0].Shape().size() != in_tensors[1].Shape().size()) {
    MS_LOG(WARNING) << name_ << " for the two inputs, the dimension size must be same."
                    << " size 1 is:" << in_tensors[0].Shape().size() << " size 2 is:" << in_tensors[1].Shape().size();
    return RET_NOT_SUPPORT;
  }
  auto type = primitive->value_type();
  if (type == mindspore::schema::PrimitiveType_Less && in_tensors[0].Shape().size() == 1) {
    MS_LOG(WARNING) << name_ << " not support input 1d";
    return RET_NOT_SUPPORT;
  }
  if (type == mindspore::schema::PrimitiveType_Equal && in_tensors[0].Shape().size() == ARITHMETIC_INPUT_NUM) {
    MS_LOG(WARNING) << name_ << " not support input 2d";
    return RET_NOT_SUPPORT;
  }
  return RET_OK;
}

template <typename T>
ge::Operator *CreateOperator(const std::string &name) {
  auto op = new (std::nothrow) T(name);
  if (op == nullptr) {
    MS_LOG(ERROR) << name << " op is nullptr";
    return nullptr;
  }
  return op;
}

int ArithmeticNPUOp::Init(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                          const std::vector<mindspore::MSTensor> &out_tensors) {
  switch (type_) {
    case schema::PrimitiveType_MulFusion:
      op_ = CreateOperator<hiai::op::Mul>(name_);
      act_type_ = primitive->value_as_MulFusion()->activation_type();
      break;
    case schema::PrimitiveType_AddFusion:
      op_ = CreateOperator<hiai::op::Add>(name_);
      act_type_ = primitive->value_as_AddFusion()->activation_type();
      break;
    case schema::PrimitiveType_SubFusion:
      op_ = CreateOperator<hiai::op::Sub>(name_);
      act_type_ = primitive->value_as_SubFusion()->activation_type();
      break;
    case schema::PrimitiveType_DivFusion:
      op_ = CreateOperator<hiai::op::RealDiv>(name_);
      act_type_ = primitive->value_as_DivFusion()->activation_type();
      break;
    case schema::PrimitiveType_FloorMod:
      op_ = CreateOperator<hiai::op::FloorMod>(name_);
      break;
    case schema::PrimitiveType_FloorDiv:
      op_ = CreateOperator<hiai::op::FloorDiv>(name_);
      break;
    case schema::PrimitiveType_LogicalAnd:
      op_ = CreateOperator<hiai::op::LogicalAnd>(name_);
      break;
    case schema::PrimitiveType_LogicalOr:
      op_ = CreateOperator<hiai::op::LogicalOr>(name_);
      break;
    case schema::PrimitiveType_Maximum:
      op_ = CreateOperator<hiai::op::Maximum>(name_);
      break;
    case schema::PrimitiveType_Minimum:
      op_ = CreateOperator<hiai::op::Minimum>(name_);
      break;
    case schema::PrimitiveType_NotEqual:
      op_ = CreateOperator<hiai::op::NotEqual>(name_);
      break;
    case schema::PrimitiveType_Equal:
      op_ = CreateOperator<hiai::op::Equal>(name_);
      break;
    case schema::PrimitiveType_Less:
      op_ = CreateOperator<hiai::op::Less>(name_);
      break;
    case schema::PrimitiveType_LessEqual:
      op_ = CreateOperator<hiai::op::LessEqual>(name_);
      break;
    case schema::PrimitiveType_Greater:
      op_ = CreateOperator<hiai::op::Greater>(name_);
      break;
    case schema::PrimitiveType_GreaterEqual:
      op_ = CreateOperator<hiai::op::GreaterEqual>(name_);
      break;
    default:
      MS_LOG(ERROR) << "Unsupported primitive type: " << schema::EnumNamePrimitiveType(type_);
      return RET_ERROR;
  }
  auto ret = SetActivation();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Arithmetic npu op set activation failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int ArithmeticNPUOp::SetActivation() {
  if (act_type_ != schema::ActivationType_NO_ACTIVATION) {
    act_ = new (std::nothrow) hiai::op::Activation(name_ + "_act");
    if (act_ == nullptr) {
      MS_LOG(ERROR) << "New activation npu operator for op " << name_ << " failed.";
      return RET_ERROR;
    }
    auto act_mode = ConverterToNPUActivationMode(act_type_);
    if (act_mode == ACTIVATION_INVALID) {
      MS_LOG(ERROR) << "Unsupported activation type for op " << name_;
      return RET_ERROR;
    }
    act_->set_attr_mode(act_mode);
    act_->set_input_x(*op_);
  }
  return RET_OK;
}

template <typename T>
void SetInputs(const std::vector<ge::Operator *> &npu_inputs, ge::Operator *op) {
  auto cur_op = reinterpret_cast<T *>(op);
  cur_op->set_input_x1(*npu_inputs[0]);
  cur_op->set_input_x2(*npu_inputs[1]);
  return;
}

int ArithmeticNPUOp::SetNPUInputs(const std::vector<mindspore::MSTensor> &in_tensors,
                                  const std::vector<mindspore::MSTensor> &out_tensors,
                                  const std::vector<ge::Operator *> &npu_inputs) {
  switch (type_) {
    case schema::PrimitiveType_MulFusion:
      SetInputs<hiai::op::Mul>(npu_inputs, op_);
      break;
    case schema::PrimitiveType_AddFusion:
      SetInputs<hiai::op::Add>(npu_inputs, op_);
      break;
    case schema::PrimitiveType_SubFusion:
      SetInputs<hiai::op::Sub>(npu_inputs, op_);
      break;
    case schema::PrimitiveType_DivFusion:
      SetInputs<hiai::op::RealDiv>(npu_inputs, op_);
      break;
    case schema::PrimitiveType_FloorMod:
      SetInputs<hiai::op::FloorMod>(npu_inputs, op_);
      break;
    case schema::PrimitiveType_FloorDiv:
      op_ = CreateOperator<hiai::op::FloorDiv>(name_);
      break;
    case schema::PrimitiveType_LogicalAnd:
      SetInputs<hiai::op::LogicalAnd>(npu_inputs, op_);
      break;
    case schema::PrimitiveType_LogicalOr:
      SetInputs<hiai::op::LogicalOr>(npu_inputs, op_);
      break;
    case schema::PrimitiveType_Maximum:
      SetInputs<hiai::op::Maximum>(npu_inputs, op_);
      break;
    case schema::PrimitiveType_Minimum:
      SetInputs<hiai::op::Minimum>(npu_inputs, op_);
      break;
    case schema::PrimitiveType_NotEqual:
      SetInputs<hiai::op::NotEqual>(npu_inputs, op_);
      break;
    case schema::PrimitiveType_Equal:
      SetInputs<hiai::op::Equal>(npu_inputs, op_);
      break;
    case schema::PrimitiveType_Less:
      SetInputs<hiai::op::Less>(npu_inputs, op_);
      break;
    case schema::PrimitiveType_LessEqual:
      SetInputs<hiai::op::LessEqual>(npu_inputs, op_);
      break;
    case schema::PrimitiveType_Greater:
      SetInputs<hiai::op::Greater>(npu_inputs, op_);
      break;
    case schema::PrimitiveType_GreaterEqual:
      SetInputs<hiai::op::GreaterEqual>(npu_inputs, op_);
      break;
    default:
      MS_LOG(ERROR) << "SetInputs for npu op " << name_ << " failed.";
      return RET_ERROR;
  }
  return RET_OK;
}

int ArithmeticNPUOp::SetNPUInputs(
  const std::vector<mindspore::MSTensor> &in_tensors, const std::vector<mindspore::MSTensor> &out_tensors,
  const std::vector<ge::Operator *> &npu_inputs,
  const std::unordered_map<int, std::pair<ge::Operator *, int>> &index2_multi_out_index) {
  auto ret = SetNPUInputs(in_tensors, out_tensors, npu_inputs);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ArithmeticNPUOp SetNPUInputs failed";
    return RET_ERROR;
  }
  if (index2_multi_out_index.empty()) {
    return RET_OK;
  }
  for (auto it : index2_multi_out_index) {
    MS_LOG(INFO) << name_ << "set input " << it.first << " from " << it.second.first << " output " << it.second.second;
    op_->SetInput(it.first, *it.second.first, it.second.second);
  }
  return RET_OK;
}

ge::Operator *ArithmeticNPUOp::GetNPUOp() {
  if (act_type_ == schema::ActivationType_NO_ACTIVATION) {
    return op_;
  }
  return act_;
}

ArithmeticNPUOp::~ArithmeticNPUOp() {
  if (op_ != nullptr) {
    delete op_;
    op_ = nullptr;
  }
  if (act_ != nullptr) {
    delete act_;
    act_ = nullptr;
  }
}
}  // namespace mindspore
