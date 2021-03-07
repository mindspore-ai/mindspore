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

#include "src/runtime/kernel/npu/arithmetic_self_npu.h"
#include <string>
#include "include/graph/op/all_ops.h"
#include "src/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH::kNPU;
using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_Ceil;
using mindspore::schema::PrimitiveType_Cos;
using mindspore::schema::PrimitiveType_Floor;
using mindspore::schema::PrimitiveType_Log;
using mindspore::schema::PrimitiveType_LogicalNot;
using mindspore::schema::PrimitiveType_Neg;
using mindspore::schema::PrimitiveType_Reciprocal;
using mindspore::schema::PrimitiveType_Round;
using mindspore::schema::PrimitiveType_Rsqrt;
using mindspore::schema::PrimitiveType_Sin;
using mindspore::schema::PrimitiveType_Sqrt;
using mindspore::schema::PrimitiveType_Square;

namespace mindspore::kernel {
int ArithmeticSelfNPUKernel::IsSupport(const std::vector<lite::Tensor *> &inputs,
                                       const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter) {
  return RET_OK;
}

template <typename T>
ge::Operator *CreateOperator(ge::Operator *input, const std::string &name) {
  auto op = new (std::nothrow) T(name);
  if (op == nullptr) {
    MS_LOG(ERROR) << name << " op is nullptr";
    return nullptr;
  }
  op->set_input_x(*input);
  return op;
}

int ArithmeticSelfNPUKernel::SetNPUInputs(const std::vector<lite::Tensor *> &inputs,
                                          const std::vector<lite::Tensor *> &outputs,
                                          const std::vector<ge::Operator *> &npu_inputs) {
  ge::Operator *op = nullptr;
  switch (op_parameter_->type_) {
    case PrimitiveType_Cos:
      op = CreateOperator<hiai::op::Cos>(npu_inputs[0], name_);
      break;
    case PrimitiveType_Log:
      op = CreateOperator<hiai::op::Log>(npu_inputs[0], name_);
      break;
    case PrimitiveType_Square:
      op = CreateOperator<hiai::op::Square>(npu_inputs[0], name_);
      break;
    case PrimitiveType_Sqrt:
      op = CreateOperator<hiai::op::Sqrt>(npu_inputs[0], name_);
      break;
    case PrimitiveType_Rsqrt:
      op = CreateOperator<hiai::op::Rsqrt>(npu_inputs[0], name_);
      break;
    case PrimitiveType_Sin:
      op = CreateOperator<hiai::op::Sin>(npu_inputs[0], name_);
      break;
    case PrimitiveType_LogicalNot:
      op = CreateOperator<hiai::op::LogicalNot>(npu_inputs[0], name_);
      break;
    case PrimitiveType_Floor:
      op = CreateOperator<hiai::op::Floor>(npu_inputs[0], name_);
      break;
    case PrimitiveType_Ceil:
      op = CreateOperator<hiai::op::Ceil>(npu_inputs[0], name_);
      break;
    case PrimitiveType_Round:
      op = CreateOperator<hiai::op::Round>(npu_inputs[0], name_);
      break;
    case PrimitiveType_Neg:
      op = CreateOperator<hiai::op::Neg>(npu_inputs[0], name_);
      break;
    case PrimitiveType_Reciprocal:
      op = CreateOperator<hiai::op::Reciprocal>(npu_inputs[0], name_);
      break;
    default:
      MS_LOG(ERROR) << "Unsupported primitive type:"
                    << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(op_parameter_->type_));
      return RET_ERROR;
  }
  if (op == nullptr) {
    MS_LOG(ERROR) << "Arithmetic self create operator return nullptr.";
    return RET_ERROR;
  }
  op_ = op;
  return RET_OK;
}

ge::Operator *mindspore::kernel::ArithmeticSelfNPUKernel::GetNPUOp() { return this->op_; }

ArithmeticSelfNPUKernel::~ArithmeticSelfNPUKernel() {
  if (op_ != nullptr) {
    delete op_;
    op_ = nullptr;
  }
}

REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_Cos, NPUKernelCreator<ArithmeticSelfNPUKernel>)
REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_Log, NPUKernelCreator<ArithmeticSelfNPUKernel>)
REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_Square, NPUKernelCreator<ArithmeticSelfNPUKernel>)
REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_Sqrt, NPUKernelCreator<ArithmeticSelfNPUKernel>)
REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_Rsqrt, NPUKernelCreator<ArithmeticSelfNPUKernel>)
REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_Sin, NPUKernelCreator<ArithmeticSelfNPUKernel>)
REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_LogicalNot, NPUKernelCreator<ArithmeticSelfNPUKernel>)
REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_Floor, NPUKernelCreator<ArithmeticSelfNPUKernel>)
REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_Ceil, NPUKernelCreator<ArithmeticSelfNPUKernel>)
REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_Round, NPUKernelCreator<ArithmeticSelfNPUKernel>)
REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_Neg, NPUKernelCreator<ArithmeticSelfNPUKernel>)
REG_KERNEL(kNPU, kNumberTypeFloat32, PrimitiveType_Reciprocal, NPUKernelCreator<ArithmeticSelfNPUKernel>)
}  // namespace mindspore::kernel
