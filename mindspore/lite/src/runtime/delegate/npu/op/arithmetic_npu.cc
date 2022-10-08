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

#include "src/runtime/delegate/npu/op/arithmetic_npu.h"
#include "src/runtime/delegate/npu/npu_converter_utils.h"
#include "src/runtime/delegate/delegate_utils.h"
#include "src/runtime/delegate/npu/transpose_kernel.h"

namespace mindspore {
constexpr int ARITHMETIC_INPUT_NUM = 2;
constexpr int MAX_HW_SIZE = 1664;
int ArithmeticNPUOp::IsSupport(const schema::Primitive *primitive, const std::vector<mindspore::MSTensor> &in_tensors,
                               const std::vector<mindspore::MSTensor> &out_tensors) {
  auto in_shape_0 = in_tensors[0].Shape();
  auto in_shape_1 = in_tensors[1].Shape();
  auto out_shape = out_tensors[0].Shape();
  // a hidden limitation in npu bottom implementation
  if (type_ == schema::PrimitiveType_MulFusion) {
    if (out_shape.size() == NPU_SHAPE_SIZE) {
      bool is_nhwc = out_tensors[0].format() == Format::NHWC;
      auto out_h = is_nhwc ? out_shape.at(NHWC_H) : out_shape.at(NCHW_H);
      auto out_w = is_nhwc ? out_shape.at(NHWC_W) : out_shape.at(NCHW_W);
      // two inputs have different shape with the output, which means both of them need broadcast
      if (in_shape_0 != out_shape && in_shape_1 != out_shape && out_h * out_w > MAX_HW_SIZE) {
        MS_LOG(WARNING) << "The size of out_height * out_width is larger than the max value (1664) that npu supports "
                           "during broadcasting.";
        return RET_NOT_SUPPORT;
      }
    }
  } else {
    if (in_shape_0.size() != 0 && in_shape_1.size() != 0 && in_shape_0.size() != in_shape_1.size()) {
      MS_LOG(WARNING) << name_
                      << " for the two inputs, the dimension size must be same. size 1 is: " << in_shape_0.size()
                      << " size 2 is: " << in_shape_1.size();
      return RET_NOT_SUPPORT;
    }
  }
  if (type_ == mindspore::schema::PrimitiveType_Less && in_shape_0.size() == 1) {
    MS_LOG(WARNING) << name_ << " dose not support 1d input.";
    return RET_NOT_SUPPORT;
  }
  if (type_ == mindspore::schema::PrimitiveType_Equal && in_shape_0.size() == ARITHMETIC_INPUT_NUM) {
    MS_LOG(WARNING) << name_ << " dose not support 2d input.";
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

int ArithmeticNPUOp::HandleAxisAndConstantInputs(std::vector<mindspore::MSTensor *> *all_tensors) {
  for (size_t i = 0; i < inputs_.size(); i++) {
    auto in_tensor = inputs_.at(i);
    if (!in_tensor.IsConst() || in_tensor.Shape().size() != DIMENSION_4D) {
      continue;
    }
    auto shape = in_tensor.Shape();
    auto new_shape = {in_tensor.Shape().at(NHWC_N), in_tensor.Shape().at(NHWC_C), in_tensor.Shape().at(NHWC_H),
                      in_tensor.Shape().at(NHWC_W)};
    auto nh2nc_tensor =
      mindspore::MSTensor::CreateTensor(in_tensor.Name() + "_nh2nc", in_tensor.DataType(), new_shape, nullptr, 0);
    if (nh2nc_tensor == nullptr) {
      MS_LOG(ERROR) << "New nchw tensor failed when inserting nchw2nhwc op.";
      return RET_ERROR;
    }
    auto dst_data = nh2nc_tensor->MutableData();
    MS_CHECK_TRUE_RET(dst_data != nullptr, RET_ERROR);
    // transpose dst_data to nchw.
    PackNHWCToNCHWFp32(in_tensor.MutableData(), dst_data, shape[NHWC_N], shape[NHWC_H] * shape[NHWC_W], shape[NHWC_C]);
    nh2nc_tensor->SetFormat(NCHW);
    inputs_.at(i) = *nh2nc_tensor;
    all_tensors->push_back(nh2nc_tensor);
  }
  return RET_OK;
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
