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

#include "src/litert/kernel/cpu/fp16/arithmetic_fp16.h"
#include "src/litert/kernel_registry.h"
#include "src/litert/kernel/cpu/fp16/common_fp16.h"
#include "nnacl/fp16/arithmetic_fp16.h"
#include "nnacl/fp16/cast_fp16.h"

using mindspore::kernel::KERNEL_ARCH;
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
using mindspore::schema::PrimitiveType_RealDiv;
using mindspore::schema::PrimitiveType_SquaredDifference;
using mindspore::schema::PrimitiveType_SubFusion;

namespace mindspore::kernel {
int ArithmeticFP16CPUKernel::ReSize() {
  in_data_size_ = sizeof(float16_t);
  out_data_size_ = sizeof(float16_t);
  if (param_->in_elements_num1_ != 1 && param_->in_elements_num0_ != 1) {
    if (a_matric_.is_const) {
      CHECK_NULL_RETURN(in_tensors_[FIRST_INPUT]->data());
      auto ret =
        ConvertFp32TensorToFp16(in_tensors_[FIRST_INPUT], static_cast<const lite::InnerContext *>(this->ms_context_));
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "convert fp32 to fp16 failed.";
        return ret;
      }
    }
    if (b_matric_.is_const) {
      CHECK_NULL_RETURN(in_tensors_[SECOND_INPUT]->data());
      auto ret =
        ConvertFp32TensorToFp16(in_tensors_[SECOND_INPUT], static_cast<const lite::InnerContext *>(this->ms_context_));
      if (ret != RET_OK) {
        MS_LOG(ERROR) << "convert fp32 to fp16 failed.";
        return ret;
      }
    }
  }
  return ArithmeticBaseCPUKernel::ReSize();
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
    {PrimitiveType_RealDiv, schema::ActivationType_RELU, ElementDivReluFp16, ElementOptDivReluFp16},
    {PrimitiveType_RealDiv, schema::ActivationType_RELU6, ElementDivRelu6Fp16, ElementOptDivRelu6Fp16},
    {PrimitiveType_RealDiv, schema::ActivationType_NO_ACTIVATION, ElementDivFp16, ElementOptDivFp16},
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
      arithmetic_run_fp16_ = fun_table[i].func_;
      arithmetic_opt_run_fp16_ = fun_table[i].opt_func_;
      return;
    }
  }
}

void ArithmeticFP16CPUKernel::DoBroadcast(void *out_data, int input_index) {
  MS_ASSERT(input_index < static_cast<int>(in_tensors_.size()));
  auto in_data = in_tensors_[input_index]->data();
  MS_ASSERT(in_data != nullptr);
  if (input_index == FIRST_INPUT) {
    TileOneDimensionFp16(reinterpret_cast<const float16_t *>(in_data), reinterpret_cast<float16_t *>(out_data), 0,
                         param_->ndim_, param_->in_shape0_, param_->in_strides0_, param_->out_strides_,
                         param_->multiples0_);
  } else {
    TileOneDimensionFp16(reinterpret_cast<const float16_t *>(in_data), reinterpret_cast<float16_t *>(out_data), 0,
                         param_->ndim_, param_->in_shape1_, param_->in_strides1_, param_->out_strides_,
                         param_->multiples1_);
  }
}

int ArithmeticFP16CPUKernel::Run() {
  auto in0_data_type = in_tensors_[FIRST_INPUT]->data_type();
  auto in1_data_type = in_tensors_[SECOND_INPUT]->data_type();
  if ((in0_data_type != kNumberTypeFloat16 && in0_data_type != kNumberTypeFloat32) ||
      (in1_data_type != kNumberTypeFloat16 && in1_data_type != kNumberTypeFloat32)) {
    MS_LOG(ERROR) << "The dataTypes of input0 and input1 should be any of float16 and float32, now input0: "
                  << in0_data_type << " and input1: " << in1_data_type;
    return RET_ERROR;
  }
  if (!a_matric_.is_valid) {
    a_matric_.data =
      ConvertInputFp32toFp16(in_tensors_[FIRST_INPUT], static_cast<const lite::InnerContext *>(this->ms_context_));
    if (in0_data_type == kNumberTypeFloat32 && a_matric_.data != nullptr) {
      fp16_buffer_.push_back(a_matric_.data);
    }
  }
  if (!b_matric_.is_valid) {
    b_matric_.data =
      ConvertInputFp32toFp16(in_tensors_[SECOND_INPUT], static_cast<const lite::InnerContext *>(this->ms_context_));
    if (in1_data_type == kNumberTypeFloat32 && b_matric_.data != nullptr) {
      fp16_buffer_.push_back(b_matric_.data);
    }
  }
  c_matric_.data = MallocOutputFp16(out_tensors_.front(), static_cast<const lite::InnerContext *>(this->ms_context_));
  auto out_data_type = out_tensors_.front()->data_type();
  if (out_data_type == kNumberTypeFloat32 && c_matric_.data != nullptr) {
    fp16_buffer_.push_back(c_matric_.data);
  }
  auto ret = ArithmeticBaseCPUKernel::Run();
  if (ret != RET_OK) {
    for (auto buffer : fp16_buffer_) {
      ms_context_->allocator->Free(buffer);
    }
    fp16_buffer_.clear();
    MS_LOG(ERROR) << "ArithmeticBaseCPUKernel->Run failed.";
    return RET_ERROR;
  }
  if (out_data_type == kNumberTypeFloat32) {
    Float16ToFloat32(static_cast<float16_t *>(c_matric_.data), reinterpret_cast<float *>(out_tensors_.front()->data()),
                     out_tensors_.front()->ElementsNum());
  }
  for (auto buffer : fp16_buffer_) {
    ms_context_->allocator->Free(buffer);
  }
  fp16_buffer_.clear();
  return RET_OK;
}

int ArithmeticFP16CPUKernel::DoExecute(const void *input0, const void *input1, void *output, int64_t size) {
  int ret = RET_OK;
  if (scalar_opt_) {
    CHECK_NULL_RETURN(arithmetic_opt_run_fp16_);
    ret =
      arithmetic_opt_run_fp16_(reinterpret_cast<const float16_t *>(input0), reinterpret_cast<const float16_t *>(input1),
                               reinterpret_cast<float16_t *>(output), size, param_);
  } else {
    CHECK_NULL_RETURN(arithmetic_run_fp16_);
    ret = arithmetic_run_fp16_(reinterpret_cast<const float16_t *>(input0), reinterpret_cast<const float16_t *>(input1),
                               reinterpret_cast<float16_t *>(output), size);
  }
  return ret;
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
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_RealDiv, LiteKernelCreator<ArithmeticFP16CPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_SquaredDifference, LiteKernelCreator<ArithmeticFP16CPUKernel>)
}  // namespace mindspore::kernel
