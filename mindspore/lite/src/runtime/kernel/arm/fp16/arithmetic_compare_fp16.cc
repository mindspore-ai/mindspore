/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include "src/runtime/kernel/arm/fp16/arithmetic_compare_fp16.h"
#include "src/runtime/kernel/arm/fp16/common_fp16.h"
#include "nnacl/fp16/arithmetic_fp16.h"
#include "nnacl/fp16/cast_fp16.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Equal;
using mindspore::schema::PrimitiveType_Greater;
using mindspore::schema::PrimitiveType_GreaterEqual;
using mindspore::schema::PrimitiveType_Less;
using mindspore::schema::PrimitiveType_LessEqual;
using mindspore::schema::PrimitiveType_NotEqual;

namespace mindspore::kernel {
ARITHMETIC_COMPARE_FUNC_INFO_FP16 arithmetic_cp_fun_table_fp16[] = {
  {PrimitiveType_NotEqual, schema::ActivationType_NO_ACTIVATION, ElementNotEqualFp16, ElementOptNotEqualFp16},
  {PrimitiveType_Equal, schema::ActivationType_NO_ACTIVATION, ElementEqualFp16, ElementOptEqualFp16},
  {PrimitiveType_Less, schema::ActivationType_NO_ACTIVATION, ElementLessFp16, ElementOptLessFp16},
  {PrimitiveType_LessEqual, schema::ActivationType_NO_ACTIVATION, ElementLessEqualFp16, ElementOptLessEqualFp16},
  {PrimitiveType_Greater, schema::ActivationType_NO_ACTIVATION, ElementGreaterFp16, ElementOptGreaterFp16},
  {PrimitiveType_GreaterEqual, schema::ActivationType_NO_ACTIVATION, ElementGreaterEqualFp16,
   ElementOptGreaterEqualFp16}};

ArithmeticCompareFuncFp16 GetArithmeticCompareFun(int primitive_type, int activation_type) {
  size_t length = sizeof(arithmetic_cp_fun_table_fp16) / sizeof(ARITHMETIC_COMPARE_FUNC_INFO_FP16);
  for (size_t i = 0; i < length; i++) {
    if (arithmetic_cp_fun_table_fp16[i].primitive_type_ == primitive_type &&
        arithmetic_cp_fun_table_fp16[i].activation_type_ == activation_type) {
      return arithmetic_cp_fun_table_fp16[i].func_;
    }
  }
  return nullptr;
}

ArithmeticCompareOptFuncFp16 GetOptimizedArithmeticCompareFun(int primitive_type, int activation_type) {
  size_t length = sizeof(arithmetic_cp_fun_table_fp16) / sizeof(ARITHMETIC_COMPARE_FUNC_INFO_FP16);
  for (size_t i = 0; i < length; i++) {
    if (arithmetic_cp_fun_table_fp16[i].primitive_type_ == primitive_type &&
        arithmetic_cp_fun_table_fp16[i].activation_type_ == activation_type) {
      return arithmetic_cp_fun_table_fp16[i].opt_func_;
    }
  }
  return nullptr;
}

int ArithmeticCompareFP16CPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ArithmeticCompareFP16CPUKernel::ReSize() {
  param_->in_elements_num0_ = in_tensors_.at(0)->ElementsNum();
  param_->in_elements_num1_ = in_tensors_.at(1)->ElementsNum();
  param_->out_elements_num_ = out_tensors_.at(0)->ElementsNum();

  if (param_->in_elements_num0_ == 1 || param_->in_elements_num1_ == 1) {
    param_->broadcasting_ = false;
    arithmetic_opt_func_ = GetOptimizedArithmeticCompareFun(param_->op_parameter_.type_, param_->activation_type_);
  } else {
    arithmetic_func_ = GetArithmeticCompareFun(param_->op_parameter_.type_, param_->activation_type_);
  }
  if (arithmetic_opt_func_ == nullptr && arithmetic_func_ == nullptr) {
    MS_LOG(ERROR) << "arithmetic_opt_func_ and arithmetic_func_ function is nullptr!";
    return RET_ERROR;
  }
  if (param_->broadcasting_) {
    outside_ = 1;
    for (int i = param_->ndim_ - 1; i >= 0; --i) {
      if (param_->in_shape0_[i] != param_->in_shape1_[i]) {
        break_pos_ = i;
        break;
      }
      outside_ *= param_->out_shape_[i];
    }
    ComputeStrides(param_->in_shape0_, param_->in_strides0_, param_->ndim_);
    ComputeStrides(param_->in_shape1_, param_->in_strides1_, param_->ndim_);
    ComputeStrides(param_->out_shape_, param_->out_strides_, param_->ndim_);
  }
  return RET_OK;
}

int ArithmeticCompareFP16CPUKernel::BroadcastRun(float16_t *input0, float16_t *input1, uint8_t *output, int dim,
                                                 int out_count, int cur_offset) {
  if (dim > break_pos_) {
    return arithmetic_func_(input0 + cur_offset, input1 + cur_offset, output + cur_offset, out_count);
  }
  for (int i = 0; i < param_->out_shape_[dim]; ++i) {
    int pos0 = param_->in_shape0_[dim] == 1 ? 0 : i;
    int pos1 = param_->in_shape1_[dim] == 1 ? 0 : i;
    int ret = BroadcastRun(input0 + pos0 * param_->in_strides0_[dim], input1 + pos1 * param_->in_strides1_[dim],
                           output + i * param_->out_strides_[dim], dim + 1, out_count, cur_offset);
    if (ret != RET_OK) {
      return RET_ERROR;
    }
  }
  return RET_OK;
}

int ArithmeticCompareFP16CPUKernel::DoArithmetic(int task_id) {
  int stride_per_thread = UP_DIV(param_->broadcasting_ ? outside_ : param_->out_elements_num_, context_->thread_num_);
  int cur_offset = stride_per_thread * task_id;
  int cur_count = param_->broadcasting_ ? MSMIN(stride_per_thread, outside_ - cur_offset)
                                        : MSMIN(stride_per_thread, param_->out_elements_num_ - cur_offset);
  if (cur_count <= 0) {
    return RET_OK;
  }

  int ret = RET_OK;
  if (param_->broadcasting_) {
    ret = BroadcastRun(input0_fp16_, input1_fp16_, output_fp16_, 0, cur_count, cur_offset);
  } else if (param_->in_elements_num0_ == 1) {
    ret = arithmetic_opt_func_(input0_fp16_, input1_fp16_ + cur_offset, output_fp16_ + cur_offset, cur_count, param_);
  } else if (param_->in_elements_num1_ == 1) {
    ret = arithmetic_opt_func_(input0_fp16_ + cur_offset, input1_fp16_, output_fp16_ + cur_offset, cur_count, param_);
  } else {
    ret = arithmetic_func_(input0_fp16_ + cur_offset, input1_fp16_ + cur_offset, output_fp16_ + cur_offset, cur_count);
  }
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DoArithmetic failed, ret = " << ret;
  }
  return ret;
}

static int ArithmeticsRunFp16(void *cdata, int task_id) {
  auto arithmetic_kernel = reinterpret_cast<ArithmeticCompareFP16CPUKernel *>(cdata);
  auto ret = arithmetic_kernel->DoArithmetic(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ArithmeticsRunFp16 error task_id[" << task_id << "] ret[" << ret << "]";
  }
  return ret;
}

int ArithmeticCompareFP16CPUKernel::Run() {
  auto output_tensor = out_tensors_.at(0);
  is_input0_fp32_ = in_tensors_.at(0)->data_type() == kNumberTypeFloat32;
  is_input1_fp32_ = in_tensors_.at(1)->data_type() == kNumberTypeFloat32;

  input0_fp16_ = ConvertInputFp32toFp16(in_tensors_.at(0), context_);
  input1_fp16_ = ConvertInputFp32toFp16(in_tensors_.at(1), context_);
  output_fp16_ = reinterpret_cast<uint8_t *>(output_tensor->MutableData());
  if (input0_fp16_ == nullptr || input1_fp16_ == nullptr || output_fp16_ == nullptr) {
    MS_LOG(ERROR) << "Memory allocation failed";
    FreeTmpBuffer();
    return RET_ERROR;
  }
  auto ret = ParallelLaunch(this->context_->thread_pool_, ArithmeticsRunFp16, this, context_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "ArithmeticsRunFp16 run error error_code[" << ret << "]";
  }
  FreeTmpBuffer();
  return ret;
}

void ArithmeticCompareFP16CPUKernel::FreeTmpBuffer() {
  if (is_input0_fp32_) {
    context_->allocator->Free(input0_fp16_);
    input0_fp16_ = nullptr;
  }
  if (is_input1_fp32_) {
    context_->allocator->Free(input1_fp16_);
    input1_fp16_ = nullptr;
  }
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_NotEqual, LiteKernelCreator<ArithmeticCompareFP16CPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Equal, LiteKernelCreator<ArithmeticCompareFP16CPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Less, LiteKernelCreator<ArithmeticCompareFP16CPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_LessEqual, LiteKernelCreator<ArithmeticCompareFP16CPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Greater, LiteKernelCreator<ArithmeticCompareFP16CPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_GreaterEqual, LiteKernelCreator<ArithmeticCompareFP16CPUKernel>)
}  // namespace mindspore::kernel
