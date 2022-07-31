/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#include <random>
#include "src/litert/kernel/cpu/fp16_grad/dropout_fp16_grad.h"
#include "nnacl/fp16_grad/dropout_grad.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/fp32_grad/dropout_parameter.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_DropoutGrad;

namespace mindspore::kernel {
int DropoutGradCPUKernelFp16::Prepare() {
  CHECK_NULL_RETURN(op_parameter_);
  auto param = reinterpret_cast<DropoutParameter *>(op_parameter_);
  if (param == nullptr) {
    MS_LOG(ERROR) << "Dropout op_parameter_ nullptr";
    return RET_NULL_PTR;
  }
  auto ratio = param->ratio_;
  if ((ratio > 1.0f) || (ratio < 0.0f)) {
    MS_LOG(ERROR) << "unsupported ratio value - Dropout ratio should be between zero to one";
    return RET_ERROR;
  }
  if (ratio >= 1.0f) {
    scale_ = 1.0f;
  } else {
    scale_ = 1. / (1. - ratio);
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }

  CHECK_LESS_RETURN(in_tensors_.size(), THIRD_INPUT);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(in_tensors_.at(0));
  CHECK_NULL_RETURN(in_tensors_.at(1));
  CHECK_NULL_RETURN(out_tensors_.at(0));
  return ReSize();
}

int DropoutGradCPUKernelFp16::ReSize() { return RET_OK; }

int DropoutGradCPUKernelFp16::DoExecute(int task_id) {
  auto yt_ptr = reinterpret_cast<float16_t *>(in_tensors_.at(kInputIndex)->data());
  auto mask_ptr = reinterpret_cast<float16_t *>(in_tensors_.at(1)->data());
  auto output_ptr = reinterpret_cast<float16_t *>(out_tensors_.at(kOutputIndex)->data());
  CHECK_NULL_RETURN(yt_ptr);
  CHECK_NULL_RETURN(mask_ptr);
  CHECK_NULL_RETURN(output_ptr);
  auto length = in_tensors_.at(kInputIndex)->ElementsNum();
  int stride = UP_DIV(length, thread_count_);
  int count = MSMIN(stride, length - stride * task_id);
  if (count > 0) {
    int start = stride * task_id;
    DropoutFp16Grad(&(yt_ptr[start]), &(mask_ptr[start]), &(output_ptr[start]), count, (float16_t)scale_);
  }
  return RET_OK;
}

int RunDropoutFp16Grad(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto dropout = reinterpret_cast<DropoutGradCPUKernelFp16 *>(cdata);
  CHECK_NULL_RETURN(dropout);
  auto error_code = dropout->DoExecute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Dropout Grad Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int DropoutGradCPUKernelFp16::Run() {
  int error_code = ParallelLaunch(this->ms_context_, RunDropoutFp16Grad, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Dropout Grad function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_DropoutGrad, LiteKernelCreator<DropoutGradCPUKernelFp16>)
}  // namespace mindspore::kernel
