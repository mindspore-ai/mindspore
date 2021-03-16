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
int ArithmeticCompareCPUKernel::BroadcastRun(void *input0, void *input1, void *output, int dim, int out_count,
                                             int out_thread_stride) {
  if (dim > break_pos_) {
    if (in_tensors_[0]->data_type() == kNumberTypeInt || in_tensors_[0]->data_type() == kNumberTypeInt32) {
      return func_int32_(reinterpret_cast<int *>(input0) + out_thread_stride,
                         reinterpret_cast<int *>(input1) + out_thread_stride,
                         reinterpret_cast<uint8_t *>(output) + out_thread_stride, out_count);
    }
    return func_fp32_(reinterpret_cast<float *>(input0) + out_thread_stride,
                      reinterpret_cast<float *>(input1) + out_thread_stride,
                      reinterpret_cast<uint8_t *>(output) + out_thread_stride, out_count);
  }
  for (int i = 0; i < param_->out_shape_[dim]; ++i) {
    int pos0_ = param_->in_shape0_[dim] == 1 ? 0 : i;
    int pos1_ = param_->in_shape1_[dim] == 1 ? 0 : i;
    int error_code;
    if (in_tensors_[0]->data_type() == kNumberTypeInt || in_tensors_[0]->data_type() == kNumberTypeInt32) {
      error_code = BroadcastRun(reinterpret_cast<int *>(input0) + pos0_ * param_->in_strides0_[dim],
                                reinterpret_cast<int *>(input1) + pos1_ * param_->in_strides1_[dim],
                                reinterpret_cast<uint8_t *>(output) + i * param_->out_strides_[dim], dim + 1, out_count,
                                out_thread_stride);
    } else {
      error_code = BroadcastRun(reinterpret_cast<float *>(input0) + pos0_ * param_->in_strides0_[dim],
                                reinterpret_cast<float *>(input1) + pos1_ * param_->in_strides1_[dim],
                                reinterpret_cast<uint8_t *>(output) + i * param_->out_strides_[dim], dim + 1, out_count,
                                out_thread_stride);
    }
    if (error_code != RET_OK) {
      return error_code;
    }
  }
  return RET_OK;
}

int ArithmeticCompareCPUKernel::DoArithmetic(int task_id) {
  auto element_num = out_tensors_[0]->ElementsNum();

  MS_ASSERT(context_->thread_num_ != 0);
  int stride = UP_DIV(element_num, context_->thread_num_);
  int count = MSMIN(stride, element_num - stride * task_id);
  if (count <= 0) {
    return RET_OK;
  }

  if (func_fp32_ == nullptr) {
    MS_LOG(ERROR) << "func_fp32_ function is nullptr!";
    return RET_ERROR;
  }

  int error_code;
  if (param_->broadcasting_) {  // need broadcast
    stride = UP_DIV(outside_, context_->thread_num_);
    int out_count = MSMIN(stride, outside_ - stride * task_id);
    int out_thread_stride = stride * task_id;
    if (out_count <= 0) {
      return RET_OK;
    }
    if (in_tensors_[0]->data_type() == kNumberTypeFloat32) {
      error_code =
        BroadcastRun(reinterpret_cast<float *>(input0_ptr_), reinterpret_cast<float *>(input1_ptr_),
                     reinterpret_cast<uint8_t *>(out_tensors_[0]->data_c()), 0, out_count, out_thread_stride);
    } else {
      error_code =
        BroadcastRun(reinterpret_cast<int *>(input0_ptr_), reinterpret_cast<int *>(input1_ptr_),
                     reinterpret_cast<uint8_t *>(out_tensors_[0]->data_c()), 0, out_count, out_thread_stride);
    }
  } else {  // no broadcast, neither is scalar, two same shape
    if (in_tensors_[0]->data_type() == kNumberTypeFloat32) {
      error_code = func_fp32_(reinterpret_cast<float *>(input0_ptr_) + stride * task_id,
                              reinterpret_cast<float *>(input1_ptr_) + stride * task_id,
                              reinterpret_cast<uint8_t *>(out_tensors_[0]->data_c()) + stride * task_id, count);
    } else {
      error_code = func_int32_(reinterpret_cast<int *>(input0_ptr_) + stride * task_id,
                               reinterpret_cast<int *>(input1_ptr_) + stride * task_id,
                               reinterpret_cast<uint8_t *>(out_tensors_[0]->data_c()) + stride * task_id, count);
    }
  }
  if (error_code != RET_OK) {
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Equal, LiteKernelCreator<ArithmeticCompareCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Equal, LiteKernelCreator<ArithmeticCompareCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_NotEqual, LiteKernelCreator<ArithmeticCompareCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Less, LiteKernelCreator<ArithmeticCompareCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Less, LiteKernelCreator<ArithmeticCompareCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_LessEqual, LiteKernelCreator<ArithmeticCompareCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_LessEqual, LiteKernelCreator<ArithmeticCompareCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Greater, LiteKernelCreator<ArithmeticCompareCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_GreaterEqual, LiteKernelCreator<ArithmeticCompareCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_GreaterEqual, LiteKernelCreator<ArithmeticCompareCPUKernel>)
}  // namespace mindspore::kernel
