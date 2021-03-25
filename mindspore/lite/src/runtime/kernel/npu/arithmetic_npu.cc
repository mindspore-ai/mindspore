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

#include "src/runtime/kernel/npu/arithmetic_npu.h"
#include <string>
#include "include/graph/op/all_ops.h"
#include "src/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH::kNPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::ActivationType_NO_ACTIVATION;
using mindspore::schema::ActivationType_RELU;
using mindspore::schema::ActivationType_RELU6;
using mindspore::schema::PrimitiveType_AddFusion;
using mindspore::schema::PrimitiveType_DivFusion;
using mindspore::schema::PrimitiveType_Equal;
using mindspore::schema::PrimitiveType_FloorDiv;
using mindspore::schema::PrimitiveType_FloorMod;
using mindspore::schema::PrimitiveType_Greater;
using mindspore::schema::PrimitiveType_GreaterEqual;
using mindspore::schema::PrimitiveType_Less;
using mindspore::schema::PrimitiveType_LessEqual;
using mindspore::schema::PrimitiveType_LogicalAnd;
using mindspore::schema::PrimitiveType_LogicalOr;
using mindspore::schema::PrimitiveType_Maximum;
using mindspore::schema::PrimitiveType_Minimum;
using mindspore::schema::PrimitiveType_MulFusion;
using mindspore::schema::PrimitiveType_NotEqual;
using mindspore::schema::PrimitiveType_SubFusion;

namespace mindspore::kernel {
int ArithmeticNPUKernel::IsSupport(const std::vector<lite::Tensor *> &inputs,
                                   const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter) {
  if (inputs[0]->shape() != inputs[1]->shape()) {
    MS_LOG(WARNING) << name_ << " for the two inputs, the corresponding dimensions must have the same value."
                    << " shape 1 is:" << inputs[0]->shape() << " shape 2 is:" << inputs[1]->shape();
    return RET_ERROR;
  }
  auto type = static_cast<schema::PrimitiveType>(opParameter->type_);
  if (type == mindspore::schema::PrimitiveType_Less && inputs[0]->shape().size() == 1) {
    MS_LOG(WARNING) << name_ << " not support input 1d";
    return RET_ERROR;
  }
  if (type == mindspore::schema::PrimitiveType_Equal && inputs[0]->shape().size() == 2) {
    MS_LOG(WARNING) << name_ << " not support input 2d";
    return RET_ERROR;
  }
  return RET_OK;
}

template <typename T>
ge::Operator *CreateOperator(const std::vector<ge::Operator *> &npu_inputs, const std::string &name) {
  auto op = new (std::nothrow) T(name);
  if (op == nullptr) {
    MS_LOG(ERROR) << name << " op is nullptr";
    return nullptr;
  }
  op->set_input_x1(*npu_inputs[0]);
  op->set_input_x2(*npu_inputs[1]);
  return op;
}

int ArithmeticNPUKernel::SetActivation() {
  if (activation_type_ != ActivationType_NO_ACTIVATION) {
    act_ = new (std::nothrow) hiai::op::Activation(name_ + "_act");
    if (act_ == nullptr) {
      MS_LOG(ERROR) << "New activation npu operator for op " << name_ << " failed.";
      return RET_ERROR;
    }
    act_->set_input_x(*op_);
    if (activation_type_ == ActivationType_RELU) {
      act_->set_attr_mode(1);
    } else if (activation_type_ == ActivationType_RELU6) {
      act_->set_attr_mode(14);
    } else {
      MS_LOG(ERROR) << "Unsupported activation type for op " << name_;
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int ArithmeticNPUKernel::SetNPUInputs(const std::vector<lite::Tensor *> &inputs,
                                      const std::vector<lite::Tensor *> &outputs,
                                      const std::vector<ge::Operator *> &npu_inputs) {
  ge::Operator *op = nullptr;
  switch (op_parameter_->type_) {
    case PrimitiveType_MulFusion:
      op = CreateOperator<hiai::op::Mul>(npu_inputs, name_);
      break;
    case PrimitiveType_AddFusion:
      op = CreateOperator<hiai::op::Add>(npu_inputs, name_);
      break;
    case PrimitiveType_SubFusion:
      op = CreateOperator<hiai::op::Sub>(npu_inputs, name_);
      break;
    case PrimitiveType_DivFusion:
      op = CreateOperator<hiai::op::RealDiv>(npu_inputs, name_);
      break;
    case PrimitiveType_FloorMod:
      op = CreateOperator<hiai::op::FloorMod>(npu_inputs, name_);
      break;
    case PrimitiveType_FloorDiv:
      op = CreateOperator<hiai::op::FloorDiv>(npu_inputs, name_);
      break;
    case PrimitiveType_LogicalAnd:
      op = CreateOperator<hiai::op::LogicalAnd>(npu_inputs, name_);
      break;
    case PrimitiveType_LogicalOr:
      op = CreateOperator<hiai::op::LogicalOr>(npu_inputs, name_);
      break;
    case PrimitiveType_Maximum:
      op = CreateOperator<hiai::op::Maximum>(npu_inputs, name_);
      break;
    case PrimitiveType_Minimum:
      op = CreateOperator<hiai::op::Minimum>(npu_inputs, name_);
      break;
    case PrimitiveType_NotEqual:
      op = CreateOperator<hiai::op::NotEqual>(npu_inputs, name_);
      break;
    case PrimitiveType_Equal:
      op = CreateOperator<hiai::op::Equal>(npu_inputs, name_);
      break;
    case PrimitiveType_Less:
      op = CreateOperator<hiai::op::Less>(npu_inputs, name_);
      break;
    case PrimitiveType_LessEqual:
      op = CreateOperator<hiai::op::LessEqual>(npu_inputs, name_);
      break;
    case PrimitiveType_Greater:
      op = CreateOperator<hiai::op::Greater>(npu_inputs, name_);
      break;
    case PrimitiveType_GreaterEqual:
      op = CreateOperator<hiai::op::GreaterEqual>(npu_inputs, name_);
      break;
    default:
      MS_LOG(ERROR) << "Unsupported primitive type:"
                    << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(op_parameter_->type_));
      return RET_ERROR;
  }
  if (op == nullptr) {
    MS_LOG(ERROR) << "Arithmetic create operator return nullptr.";
    return RET_ERROR;
  }
  op_ = op;

  auto ret = SetActivation();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Arithmetic npu op set activation failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

ge::Operator *mindspore::kernel::ArithmeticNPUKernel::GetNPUOp() {
  if (activation_type_ == ActivationType_NO_ACTIVATION) {
    return op_;
  }
  return act_;
}

ArithmeticNPUKernel::~ArithmeticNPUKernel() {
  if (op_ != nullptr) {
    delete op_;
    op_ = nullptr;
  }
  if (act_ != nullptr) {
    delete act_;
    act_ = nullptr;
  }
}

// SquaredDifference don't supported in NPU.
REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_MulFusion, NPUKernelCreator<ArithmeticNPUKernel>)
REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_AddFusion, NPUKernelCreator<ArithmeticNPUKernel>)
REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_SubFusion, NPUKernelCreator<ArithmeticNPUKernel>)
REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_DivFusion, NPUKernelCreator<ArithmeticNPUKernel>)
REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_FloorMod, NPUKernelCreator<ArithmeticNPUKernel>)
REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_FloorDiv, NPUKernelCreator<ArithmeticNPUKernel>)
REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_LogicalAnd, NPUKernelCreator<ArithmeticNPUKernel>)
REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_LogicalOr, NPUKernelCreator<ArithmeticNPUKernel>)
REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_Maximum, NPUKernelCreator<ArithmeticNPUKernel>)
REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_Minimum, NPUKernelCreator<ArithmeticNPUKernel>)
REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_NotEqual, NPUKernelCreator<ArithmeticNPUKernel>)
REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_Equal, NPUKernelCreator<ArithmeticNPUKernel>)
REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_Less, NPUKernelCreator<ArithmeticNPUKernel>)
REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_LessEqual, NPUKernelCreator<ArithmeticNPUKernel>)
REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_Greater, NPUKernelCreator<ArithmeticNPUKernel>)
REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_GreaterEqual, NPUKernelCreator<ArithmeticNPUKernel>)
}  // namespace mindspore::kernel
