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
#include "src/runtime/kernel/arm/fp16/arithmetic_fp16.h"
#include "src/runtime/kernel/arm/fp16/common_fp16.h"
#include "src/kernel_registry.h"
#include "nnacl/fp16/cast_fp16.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

using mindspore::schema::PrimitiveType_AddFusion;
using mindspore::schema::PrimitiveType_DivFusion;
using mindspore::schema::PrimitiveType_Eltwise;
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
using mindspore::schema::PrimitiveType_SquaredDifference;
using mindspore::schema::PrimitiveType_SubFusion;

namespace mindspore::kernel {
int ArithmeticFP16CPUKernel::ReSize() {
  auto ret = ArithmeticCPUKernel::ReSize();
  data_type_len_ = sizeof(float16_t);
  return ret;
}

int ArithmeticFP16CPUKernel::CheckDataType() {
  auto in0_dataType = in_tensors_.at(0)->data_type();
  auto in1_dataType = in_tensors_.at(1)->data_type();
  if ((in0_dataType != kNumberTypeFloat16 && in0_dataType != kNumberTypeFloat32) ||
      (in1_dataType != kNumberTypeFloat16 && in1_dataType != kNumberTypeFloat32)) {
    MS_LOG(ERROR)
      << "The dataTypes of input tensor0 and input tensor1 should be any of float16 and float32, otherwise got error.";
    return RET_ERROR;
  }
  return RET_OK;
}

bool ArithmeticFP16CPUKernel::IsScalarClac() {  // 2 32 240 240, 1 1 1 1
  if ((param_->in_elements_num0_ == 1 || param_->in_elements_num1_ == 1) && (arithmetic_opt_func_ != nullptr)) {
    return true;
  } else {
    return false;
  }
}

bool ArithmeticFP16CPUKernel::IsBatchScalarCalc() {
  if (arithmetic_opt_func_ == nullptr) {
    return false;
  }
  size_t break_axis = 0;
  for (size_t i = 0; i < param_->ndim_; i++) {
    if (param_->in_shape0_[i] != param_->in_shape1_[i]) {
      break_axis = i;
      break;
    }
  }
  if (break_axis < param_->ndim_) {
    for (size_t i = break_axis; i < param_->ndim_; i++) {
      if (param_->in_shape1_[i] != 1) {
        return false;
      }
    }
  }
  break_pos_ = break_axis;
  return true;
}

void ArithmeticFP16CPUKernel::InitRunFunction(int primitive_type) {
  ARITHMETIC_FUNC_INFO_FP16 fun_table[] = {
    {PrimitiveType_MulFusion, schema::ActivationType_RELU, ElementMulReluFp16, ElementOptMulReluFp16},
    {PrimitiveType_MulFusion, schema::ActivationType_RELU6, ElementMulRelu6Fp16, ElementOptMulRelu6Fp16},
    {PrimitiveType_MulFusion, schema::ActivationType_NO_ACTIVATION, ElementMulFp16, ElementOptMulFp16},
    {PrimitiveType_AddFusion, schema::ActivationType_RELU, ElementAddReluFp16, ElementOptAddReluFp16},
    {PrimitiveType_AddFusion, schema::ActivationType_RELU6, ElementAddRelu6Fp16, ElementOptAddRelu6Fp16},
    {PrimitiveType_AddFusion, schema::ActivationType_NO_ACTIVATION, ElementAddFp16, ElementOptAddFp16},
    {PrimitiveType_SubFusion, schema::ActivationType_RELU, ElementSubReluFp16, ElementOptSubReluFp16},
    {PrimitiveType_SubFusion, schema::ActivationType_RELU6, ElementSubRelu6Fp16, ElementOptSubRelu6Fp16},
    {PrimitiveType_SubFusion, schema::ActivationType_NO_ACTIVATION, ElementSubFp16, ElementOptSubFp16},
    {PrimitiveType_DivFusion, schema::ActivationType_RELU, ElementDivReluFp16, ElementOptDivReluFp16},
    {PrimitiveType_DivFusion, schema::ActivationType_RELU6, ElementDivRelu6Fp16, ElementOptDivRelu6Fp16},
    {PrimitiveType_DivFusion, schema::ActivationType_NO_ACTIVATION, ElementDivFp16, ElementOptDivFp16},
    {PrimitiveType_FloorMod, schema::ActivationType_NO_ACTIVATION, ElementFloorModFp16, ElementOptFloorModFp16},
    {PrimitiveType_FloorDiv, schema::ActivationType_NO_ACTIVATION, ElementFloorDivFp16, ElementOptFloorDivFp16},
    {PrimitiveType_LogicalAnd, schema::ActivationType_NO_ACTIVATION, ElementLogicalAndFp16, ElementOptLogicalAndFp16},
    {PrimitiveType_LogicalOr, schema::ActivationType_NO_ACTIVATION, ElementLogicalOrFp16, ElementOptLogicalOrFp16},
    {PrimitiveType_SquaredDifference, schema::ActivationType_NO_ACTIVATION, ElementSquaredDifferenceFp16,
     ElementOptSquaredDifferenceFp16},
    {PrimitiveType_Maximum, schema::ActivationType_NO_ACTIVATION, ElementMaximumFp16, ElementOptMaximumFp16},
    {PrimitiveType_Minimum, schema::ActivationType_NO_ACTIVATION, ElementMinimumFp16, ElementOptMinimumFp16}};

  size_t length = sizeof(fun_table) / sizeof(ARITHMETIC_FUNC_INFO_FP16);
  for (size_t i = 0; i < length; i++) {
    if (fun_table[i].primitive_type_ == primitive_type && fun_table[i].activation_type_ == param_->activation_type_) {
      arithmetic_opt_func_ = fun_table[i].opt_func_;
      arithmetic_func_ = fun_table[i].func_;
      return;
    }
  }
}

int ArithmeticFP16CPUKernel::ConstTensorBroadCast() {
  int ret;
  if (in_tensors_[0]->data_c() != nullptr) {
    ret = ConvertFp32TensorToFp16(in_tensors_[0], context_);
    if (ret != RET_OK) {
      return ret;
    }
  }
  if (in_tensors_[1]->data_c() != nullptr) {
    ret = ConvertFp32TensorToFp16(in_tensors_[1], context_);
    if (ret != RET_OK) {
      return ret;
    }
  }
  return ArithmeticCPUKernel::ConstTensorBroadCast();
}

void ArithmeticFP16CPUKernel::TileConstTensor(const void *in_data, void *out_data, size_t ndim, const int *in_shape,
                                              const int *in_strides, const int *out_strides, const int *multiple) {
  TileOneDimensionFp16(reinterpret_cast<const float16_t *>(in_data), reinterpret_cast<float16_t *>(out_data), 0, ndim,
                       in_shape, in_strides, out_strides, multiple);
}

int ArithmeticFP16CPUKernel::Execute(const void *input0, const void *input1, void *output, int size, bool is_opt) {
  int ret = RET_OK;
  if (is_opt) {
    CHECK_NULL_RETURN(arithmetic_opt_func_, RET_ERROR);
    ret = arithmetic_opt_func_(reinterpret_cast<const float16_t *>(input0), reinterpret_cast<const float16_t *>(input1),
                               reinterpret_cast<float16_t *>(output), size, param_);
  } else {
    CHECK_NULL_RETURN(arithmetic_func_, RET_ERROR);
    ret = arithmetic_func_(reinterpret_cast<const float16_t *>(input0), reinterpret_cast<const float16_t *>(input1),
                           reinterpret_cast<float16_t *>(output), size);
  }
  return ret;
}

int ArithmeticFP16CPUKernel::Run() {
  if (CheckDataType() != RET_OK) {
    MS_LOG(ERROR) << "ArithmeticFP16CPUKernel check dataType failed.";
    return RET_ERROR;
  }
  if (!input0_broadcast_) {
    input0_ptr_ = ConvertInputFp32toFp16(in_tensors_.at(0), context_);
  }
  if (!input1_broadcast_) {
    input1_ptr_ = ConvertInputFp32toFp16(in_tensors_.at(1), context_);
  }
  auto output_tensor = out_tensors_.at(0);
  output_ptr_ = MallocOutputFp16(output_tensor, context_);
  if (input0_ptr_ == nullptr || input1_ptr_ == nullptr || output_ptr_ == nullptr) {
    FreeFp16Buffer();
    return RET_ERROR;
  }
  auto ret = ParallelLaunch(this->context_->thread_pool_, ArithmeticsRun, this, context_->thread_num_);
  if (out_tensors_.at(0)->data_type() == kNumberTypeFloat32) {
    Float16ToFloat32(static_cast<float16_t *>(output_ptr_), reinterpret_cast<float *>(output_tensor->MutableData()),
                     output_tensor->ElementsNum());
  }
  FreeFp16Buffer();
  return ret;
}

void ArithmeticFP16CPUKernel::FreeFp16Buffer() {
  if (!input0_broadcast_ && in_tensors_.at(0)->data_type() == kNumberTypeFloat32) {
    context_->allocator->Free(input0_ptr_);
    input0_ptr_ = nullptr;
  }
  if (!input1_broadcast_ && in_tensors_.at(1)->data_type() == kNumberTypeFloat32) {
    context_->allocator->Free(input1_ptr_);
    input1_ptr_ = nullptr;
  }
  if (out_tensors_.at(0)->data_type() == kNumberTypeFloat32) {
    context_->allocator->Free(output_ptr_);
    output_ptr_ = nullptr;
  }
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_MulFusion, LiteKernelCreator<ArithmeticFP16CPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_AddFusion, LiteKernelCreator<ArithmeticFP16CPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_SubFusion, LiteKernelCreator<ArithmeticFP16CPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_DivFusion, LiteKernelCreator<ArithmeticFP16CPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_FloorMod, LiteKernelCreator<ArithmeticFP16CPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_FloorDiv, LiteKernelCreator<ArithmeticFP16CPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_LogicalAnd, LiteKernelCreator<ArithmeticFP16CPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_LogicalOr, LiteKernelCreator<ArithmeticFP16CPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Maximum, LiteKernelCreator<ArithmeticFP16CPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Minimum, LiteKernelCreator<ArithmeticFP16CPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Eltwise, LiteKernelCreator<ArithmeticFP16CPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_SquaredDifference, LiteKernelCreator<ArithmeticFP16CPUKernel>)
}  // namespace mindspore::kernel
