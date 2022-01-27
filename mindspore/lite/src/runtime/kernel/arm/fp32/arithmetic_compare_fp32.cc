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
#include "src/runtime/kernel/arm/fp32/arithmetic_compare_fp32.h"
#include "src/kernel_registry.h"
#include "nnacl/fp32/arithmetic_compare_fp32.h"

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
void ArithmeticCompareCPUKernel::InitRunFunction(int primitive_type) {
  ARITHMETIC_COMEPARE_FUNC_INFO_FP32 fun_table[] = {
    {PrimitiveType_Equal, ElementEqualFp32, ElementEqualInt32, ElementOptEqualFp32, ElementOptEqualInt32},
    {PrimitiveType_NotEqual, ElementNotEqualFp32, ElementNotEqualInt32, ElementOptNotEqualFp32,
     ElementOptNotEqualInt32},
    {PrimitiveType_Less, ElementLessFp32, ElementLessInt32, ElementOptLessFp32, ElementOptLessInt32},
    {PrimitiveType_LessEqual, ElementLessEqualFp32, ElementLessEqualInt32, ElementOptLessEqualFp32,
     ElementOptLessEqualInt32},
    {PrimitiveType_Greater, ElementGreaterFp32, ElementGreaterInt32, ElementOptGreaterFp32, ElementOptGreaterInt32},
    {PrimitiveType_GreaterEqual, ElementGreaterEqualFp32, ElementGreaterEqualInt32, ElementOptGreaterEqualFp32,
     ElementOptGreaterEqualInt32}};
  size_t length = sizeof(fun_table) / sizeof(ARITHMETIC_COMEPARE_FUNC_INFO_FP32);
  for (size_t i = 0; i < length; i++) {
    if (fun_table[i].primitive_type_ == primitive_type) {
      func_fp32_ = fun_table[i].func_;
      func_int32_ = fun_table[i].int_func_;
      opt_func_fp32_ = fun_table[i].opt_func_;
      opt_func_int32_ = fun_table[i].opt_int_func_;
      return;
    }
  }
}

int ArithmeticCompareCPUKernel::Execute(const void *input0, const void *input1, void *output, int size, bool is_opt) {
  int ret = RET_OK;
  if (in_tensors_[0]->data_type() == kNumberTypeFloat32) {
    if (is_opt) {
      CHECK_NULL_RETURN(opt_func_fp32_);
      ret = opt_func_fp32_(reinterpret_cast<const float *>(input0), reinterpret_cast<const float *>(input1),
                           reinterpret_cast<uint8_t *>(output), size, param_);
    } else {
      CHECK_NULL_RETURN(func_fp32_);
      ret = func_fp32_(reinterpret_cast<const float *>(input0), reinterpret_cast<const float *>(input1),
                       reinterpret_cast<uint8_t *>(output), size);
    }
  } else if (in_tensors_[0]->data_type() == kNumberTypeInt || in_tensors_[0]->data_type() == kNumberTypeInt32) {
    if (is_opt) {
      CHECK_NULL_RETURN(opt_func_int32_);
      ret = opt_func_int32_(reinterpret_cast<const int *>(input0), reinterpret_cast<const int *>(input1),
                            reinterpret_cast<uint8_t *>(output), size, param_);
    } else {
      CHECK_NULL_RETURN(func_int32_);
      ret = func_int32_(reinterpret_cast<const int *>(input0), reinterpret_cast<const int *>(input1),
                        reinterpret_cast<uint8_t *>(output), size);
    }
  } else {
    MS_LOG(ERROR) << "Error Operator type " << kNumberTypeInt32;
    return RET_ERROR;
  }
  return ret;
}

int ArithmeticCompareCPUKernel::CalcArithmeticByBatch(int task_id) {
  if (break_pos_ > ARITHMETIC_SUPPORT_DIMS_NUM || param_->out_strides_[break_pos_ - 1] == 0) {
    MS_LOG(ERROR) << "param_->out_strides_[break_pos_ - 1] is 0 or break_pos_ is > 10";
    return RET_ERROR;
  }

  int batch_per_thread = UP_DIV(out_batch_, op_parameter_->thread_num_);
  int start_batch = batch_per_thread * task_id;
  int end_batch = MSMIN(start_batch + batch_per_thread, out_batch_);
  int ret = RET_ERROR;
  for (int i = start_batch; i < end_batch; i++) {
    batch_a_ptr_ = static_cast<uint8_t *>(input0_ptr_) + a_offset_[i] * a_stride_size_ * data_type_len_;
    batch_b_ptr_ = static_cast<uint8_t *>(input1_ptr_) + b_offset_[i] * b_stride_size_ * data_type_len_;
    batch_c_ptr_ = static_cast<uint8_t *>(output_ptr_) + i * c_stride_size_ * sizeof(uint8_t);
    if (batch_scalar_) {
      ret = Execute(batch_a_ptr_, batch_b_ptr_, batch_c_ptr_, c_stride_size_, true);
    } else {
      ret = Execute(batch_a_ptr_, batch_b_ptr_, batch_c_ptr_, c_stride_size_, false);
    }
    if (ret != RET_OK) {
      MS_LOG(ERROR) << "failed to calculate.";
      return RET_ERROR;
    }
  }
  return ret;
}

int ArithmeticCompareCPUKernel::DoArithmetic(int task_id) {
  if (split_by_batch_) {
    return CalcArithmeticByBatch(task_id);
  }

  int64_t element_num = out_tensors_[0]->ElementsNum();
  auto ret = RET_ERROR;
  int stride = UP_DIV(element_num, op_parameter_->thread_num_);
  int count = MSMIN(stride, element_num - stride * task_id);
  if (count <= 0) {
    return RET_OK;
  }
  CHECK_LESS_RETURN(ARITHMETIC_SUPPORT_DIMS_NUM, param_->ndim_);
  int in_offset = stride * task_id * data_type_len_;
  int out_offset = stride * task_id * sizeof(uint8_t);
  if (scalar_) {
    if (param_->in_elements_num0_ == 1) {
      ret = Execute(batch_a_ptr_, batch_b_ptr_ + in_offset, batch_c_ptr_ + out_offset, count, true);
    } else {
      ret = Execute(batch_a_ptr_ + in_offset, batch_b_ptr_, batch_c_ptr_ + out_offset, count, true);
    }
  } else {
    ret = Execute(batch_a_ptr_ + in_offset, batch_b_ptr_ + in_offset, batch_c_ptr_ + out_offset, count, false);
  }
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Equal, LiteKernelCreator<ArithmeticCompareCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Equal, LiteKernelCreator<ArithmeticCompareCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_NotEqual, LiteKernelCreator<ArithmeticCompareCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_NotEqual, LiteKernelCreator<ArithmeticCompareCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Less, LiteKernelCreator<ArithmeticCompareCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Less, LiteKernelCreator<ArithmeticCompareCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_LessEqual, LiteKernelCreator<ArithmeticCompareCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_LessEqual, LiteKernelCreator<ArithmeticCompareCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Greater, LiteKernelCreator<ArithmeticCompareCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Greater, LiteKernelCreator<ArithmeticCompareCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_GreaterEqual, LiteKernelCreator<ArithmeticCompareCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_GreaterEqual, LiteKernelCreator<ArithmeticCompareCPUKernel>)
}  // namespace mindspore::kernel
