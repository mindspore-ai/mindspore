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

#include "src/litert/kernel/cpu/fp32/arithmetic_fp32.h"
#include "src/litert/kernel_registry.h"
#include "nnacl/fp32/arithmetic_fp32.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_AddFusion;
using mindspore::schema::PrimitiveType_DivFusion;
using mindspore::schema::PrimitiveType_Eltwise;
using mindspore::schema::PrimitiveType_FloorDiv;
using mindspore::schema::PrimitiveType_FloorMod;
using mindspore::schema::PrimitiveType_LogicalAnd;
using mindspore::schema::PrimitiveType_LogicalOr;
using mindspore::schema::PrimitiveType_Maximum;
using mindspore::schema::PrimitiveType_Minimum;
using mindspore::schema::PrimitiveType_Mod;
using mindspore::schema::PrimitiveType_MulFusion;
using mindspore::schema::PrimitiveType_RealDiv;
using mindspore::schema::PrimitiveType_SquaredDifference;
using mindspore::schema::PrimitiveType_SubFusion;

namespace mindspore::kernel {
int ArithmeticCPUKernel::ReSize() {
  in_data_size_ = static_cast<int>(lite::DataTypeSize(in_tensors_.front()->data_type()));
  out_data_size_ = static_cast<int>(lite::DataTypeSize(out_tensors_.front()->data_type()));
  MS_CHECK_TRUE_MSG(in_data_size_ != 0, lite::RET_ERROR, "in-data-size is zero, which is invalid.");
  MS_CHECK_TRUE_MSG(out_data_size_ != 0, lite::RET_ERROR, "out-data-size is zero, which is invalid.");
  return ArithmeticBaseCPUKernel::ReSize();
}

int ArithmeticCPUKernel::Run() {
  if (in_tensors_[FIRST_INPUT]->data_type() != in_tensors_[SECOND_INPUT]->data_type()) {
    MS_LOG(ERROR) << "The dataTypes of input0 and input1 should be the same. input0: "
                  << in_tensors_[FIRST_INPUT]->data_type() << " vs input1" << in_tensors_[SECOND_INPUT]->data_type();
    return RET_ERROR;
  }
  if (op_parameter_->is_train_session_) {
    in_data_size_ = static_cast<int>(lite::DataTypeSize(in_tensors_.front()->data_type()));
  }
  if (!a_matric_.is_valid) {
    a_matric_.data = in_tensors_[FIRST_INPUT]->data();
  }
  if (!b_matric_.is_valid) {
    b_matric_.data = in_tensors_[SECOND_INPUT]->data();
  }
  c_matric_.data = out_tensors_.front()->data();
  return ArithmeticBaseCPUKernel::Run();
}

void ArithmeticCPUKernel::InitRunFunction(int primitive_type) {
  ARITHMETIC_FUNC_INFO_FP32 fun_table[] = {
    {PrimitiveType_MulFusion, schema::ActivationType_RELU, ElementMulRelu, ElementMulReluInt, nullptr,
     ElementOptMulRelu, ElementOptMulReluInt, nullptr},
    {PrimitiveType_MulFusion, schema::ActivationType_RELU6, ElementMulRelu6, ElementMulRelu6Int, nullptr,
     ElementOptMulRelu6, ElementOptMulRelu6Int, nullptr},
    {PrimitiveType_MulFusion, schema::ActivationType_NO_ACTIVATION, ElementMul, ElementMulInt, nullptr, ElementOptMul,
     ElementOptMulInt, nullptr},
    {PrimitiveType_AddFusion, schema::ActivationType_RELU, ElementAddRelu, nullptr, nullptr, ElementOptAddRelu, nullptr,
     nullptr},
    {PrimitiveType_AddFusion, schema::ActivationType_RELU6, ElementAddRelu6, nullptr, nullptr, ElementOptAddRelu6,
     nullptr, nullptr},
    {PrimitiveType_AddFusion, schema::ActivationType_NO_ACTIVATION, ElementAdd, ElementAddInt, nullptr, ElementOptAdd,
     ElementOptAddInt, nullptr},
    {PrimitiveType_SubFusion, schema::ActivationType_RELU, ElementSubRelu, nullptr, nullptr, ElementOptSubRelu, nullptr,
     nullptr},
    {PrimitiveType_SubFusion, schema::ActivationType_RELU6, ElementSubRelu6, nullptr, nullptr, ElementOptSubRelu6,
     nullptr, nullptr},
    {PrimitiveType_SubFusion, schema::ActivationType_NO_ACTIVATION, ElementSub, ElementSubInt, nullptr, ElementOptSub,
     ElementOptSubInt, nullptr},
    {PrimitiveType_DivFusion, schema::ActivationType_RELU, ElementDivRelu, nullptr, nullptr, ElementOptDivRelu, nullptr,
     nullptr},
    {PrimitiveType_DivFusion, schema::ActivationType_RELU6, ElementDivRelu6, nullptr, nullptr, ElementOptDivRelu6,
     nullptr, nullptr},
    {PrimitiveType_DivFusion, schema::ActivationType_NO_ACTIVATION, ElementDiv, nullptr, nullptr, ElementOptDiv,
     ElementOptDivInt, nullptr},
    {PrimitiveType_RealDiv, schema::ActivationType_RELU, ElementDivRelu, nullptr, nullptr, ElementOptDivRelu, nullptr,
     nullptr},
    {PrimitiveType_RealDiv, schema::ActivationType_RELU6, ElementDivRelu6, nullptr, nullptr, ElementOptDivRelu6,
     nullptr, nullptr},
    {PrimitiveType_RealDiv, schema::ActivationType_NO_ACTIVATION, ElementDiv, nullptr, nullptr, ElementOptDiv,
     ElementOptDivInt, nullptr},
    {PrimitiveType_LogicalAnd, schema::ActivationType_NO_ACTIVATION, ElementLogicalAnd, ElementLogicalAndInt,
     ElementLogicalAndBool, ElementOptLogicalAnd, ElementOptLogicalAndInt, ElementOptLogicalAndBool},
    {PrimitiveType_LogicalOr, schema::ActivationType_NO_ACTIVATION, ElementLogicalOr, nullptr, ElementLogicalOrBool,
     nullptr, nullptr, ElementOptLogicalOrBool},
    {PrimitiveType_Maximum, schema::ActivationType_NO_ACTIVATION, ElementMaximum, ElementMaximumInt, nullptr,
     ElementOptMaximum, ElementOptMaximumInt, nullptr},
    {PrimitiveType_Minimum, schema::ActivationType_NO_ACTIVATION, ElementMinimum, ElementMinimumInt, nullptr,
     ElementOptMinimum, ElementOptMinimumInt, nullptr},
    {PrimitiveType_FloorMod, schema::ActivationType_NO_ACTIVATION, ElementFloorMod, ElementFloorModInt, nullptr,
     ElementOptFloorMod, ElementOptFloorModInt, nullptr},
    {PrimitiveType_FloorDiv, schema::ActivationType_NO_ACTIVATION, ElementFloorDiv, ElementFloorDivInt, nullptr,
     ElementOptFloorDiv, ElementOptFloorDivInt, nullptr},
    {PrimitiveType_Mod, schema::ActivationType_NO_ACTIVATION, ElementMod, ElementModInt, nullptr, ElementOptMod,
     ElementOptModInt, nullptr},
    {PrimitiveType_SquaredDifference, schema::ActivationType_NO_ACTIVATION, ElementSquaredDifference, nullptr, nullptr,
     ElementOptSquaredDifference, nullptr, nullptr}};

  size_t length = sizeof(fun_table) / sizeof(ARITHMETIC_FUNC_INFO_FP32);
  for (size_t i = 0; i < length; i++) {
    if (fun_table[i].primitive_type_ == primitive_type && fun_table[i].activation_type_ == param_->activation_type_) {
      arithmetic_run_fp32_ = fun_table[i].func_;
      arithmetic_run_int_ = fun_table[i].int_func_;
      arithmetic_run_bool_ = fun_table[i].bool_func_;
      arithmetic_opt_run_fp32_ = fun_table[i].opt_func_;
      arithmetic_opt_run_int_ = fun_table[i].opt_int_func_;
      arithmetic_opt_run_bool_ = fun_table[i].opt_bool_func_;
      return;
    }
  }
}

void ArithmeticCPUKernel::DoBroadcast(void *out_data, int input_index) {
  MS_ASSERT(input_index < static_cast<int>(in_tensors_.size()));
  auto in_data = in_tensors_[input_index]->data();
  MS_ASSERT(in_data != nullptr);
  if (input_index == FIRST_INPUT) {
    TileOneDimensionFp32(reinterpret_cast<const float *>(in_data), reinterpret_cast<float *>(out_data), 0,
                         param_->ndim_, param_->in_shape0_, param_->in_strides0_, param_->out_strides_,
                         param_->multiples0_);
  } else {
    TileOneDimensionFp32(reinterpret_cast<const float *>(in_data), reinterpret_cast<float *>(out_data), 0,
                         param_->ndim_, param_->in_shape1_, param_->in_strides1_, param_->out_strides_,
                         param_->multiples1_);
  }
}

int ArithmeticCPUKernel::DoExecute(const void *input0, const void *input1, void *output, int64_t size) {
  int ret = RET_OK;
  if (in_tensors_[0]->data_type() == kNumberTypeFloat32) {
    if (scalar_opt_) {
      CHECK_NULL_RETURN(arithmetic_opt_run_fp32_);
      ret = arithmetic_opt_run_fp32_(reinterpret_cast<const float *>(input0), reinterpret_cast<const float *>(input1),
                                     reinterpret_cast<float *>(output), size, param_);
    } else {
      CHECK_NULL_RETURN(arithmetic_run_fp32_);
      ret = arithmetic_run_fp32_(reinterpret_cast<const float *>(input0), reinterpret_cast<const float *>(input1),
                                 reinterpret_cast<float *>(output), size);
    }
  } else if (in_tensors_[0]->data_type() == kNumberTypeBool) {
    if (scalar_opt_) {
      CHECK_NULL_RETURN(arithmetic_opt_run_bool_);
      ret = arithmetic_opt_run_bool_(reinterpret_cast<const bool *>(input0), reinterpret_cast<const bool *>(input1),
                                     reinterpret_cast<bool *>(output), size, param_);
    } else {
      CHECK_NULL_RETURN(arithmetic_run_bool_);
      ret = arithmetic_run_bool_(reinterpret_cast<const bool *>(input0), reinterpret_cast<const bool *>(input1),
                                 reinterpret_cast<bool *>(output), size);
    }
  } else {
    if (scalar_opt_) {
      CHECK_NULL_RETURN(arithmetic_opt_run_int_);
      ret = arithmetic_opt_run_int_(reinterpret_cast<const int *>(input0), reinterpret_cast<const int *>(input1),
                                    reinterpret_cast<int *>(output), size, param_);
    } else {
      CHECK_NULL_RETURN(arithmetic_run_int_);
      ret = arithmetic_run_int_(reinterpret_cast<const int *>(input0), reinterpret_cast<const int *>(input1),
                                reinterpret_cast<int *>(output), size);
    }
  }
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_MulFusion, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_MulFusion, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_AddFusion, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_AddFusion, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_AddFusion, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_SubFusion, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_SubFusion, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_DivFusion, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_RealDiv, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Mod, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Mod, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_LogicalAnd, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_LogicalAnd, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_LogicalAnd, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_LogicalOr, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeBool, PrimitiveType_LogicalOr, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Maximum, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Minimum, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Maximum, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Minimum, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_FloorDiv, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_FloorMod, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_FloorDiv, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_FloorMod, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_SquaredDifference, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Eltwise, LiteKernelCreator<ArithmeticCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_DivFusion, LiteKernelCreator<ArithmeticCPUKernel>)
}  // namespace mindspore::kernel
