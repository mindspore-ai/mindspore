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

#include "src/runtime/kernel/arm/fp16_grad/arithmetic_fp16_grad.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "nnacl/fp16_grad/arithmetic_grad.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
constexpr static int kX1Idx = 0;
constexpr static int kX2Idx = 1;
constexpr static int kDyIdx = 2;

int ArithmeticGradCPUKernelFp16::Init() {
  CHECK_LESS_RETURN(in_tensors_.size(), FOURTH_INPUT);
  CHECK_LESS_RETURN(out_tensors_.size(), THIRD_INPUT);
  CHECK_NULL_RETURN(in_tensors_[kX1Idx]);
  CHECK_NULL_RETURN(in_tensors_[kX2Idx]);
  CHECK_NULL_RETURN(in_tensors_[kDyIdx]);
  CHECK_NULL_RETURN(out_tensors_[0]);
  CHECK_NULL_RETURN(out_tensors_[1]);
  CHECK_NULL_RETURN(arithmeticParameter_);
  return RET_OK;
}

int ArithmeticGradCPUKernelFp16::ArithmeticGradMaximum(float16_t *dy, int dy_size, float16_t *dx1, int dx1_size,
                                                       float16_t *dx2, int dx2_size) {
  auto x1 = reinterpret_cast<float16_t *>(in_tensors_[kX1Idx]->data());
  auto x2 = reinterpret_cast<float16_t *>(in_tensors_[kX2Idx]->data());
  dy = reinterpret_cast<float16_t *>(in_tensors_[kDyIdx]->data());
  CHECK_NULL_RETURN(x1);
  CHECK_NULL_RETURN(x2);
  CHECK_NULL_RETURN(dy);

  MaximumByAxesFp16(x1, x2, dy, arithmeticParameter_->in_shape0_, arithmeticParameter_->in_shape1_,
                    arithmeticParameter_->out_shape_, dx1, dx2, arithmeticParameter_->ndim_);
  return RET_OK;
}

int ArithmeticGradCPUKernelFp16::ArithmeticGradMinimum(float16_t *dy, int dy_size, float16_t *dx1, int dx1_size,
                                                       float16_t *dx2, int dx2_size) {
  auto x1 = reinterpret_cast<float16_t *>(in_tensors_[kX1Idx]->data());
  auto x2 = reinterpret_cast<float16_t *>(in_tensors_[kX2Idx]->data());
  dy = reinterpret_cast<float16_t *>(in_tensors_[kDyIdx]->data());
  CHECK_NULL_RETURN(x1);
  CHECK_NULL_RETURN(x2);
  CHECK_NULL_RETURN(dy);

  MinimumByAxesFp16(x1, x2, dy, arithmeticParameter_->in_shape0_, arithmeticParameter_->in_shape1_,
                    arithmeticParameter_->out_shape_, dx1, dx2, arithmeticParameter_->ndim_);
  return RET_OK;
}

int ArithmeticGradCPUKernelFp16::ReSize() { return RET_OK; }

int ArithmeticGradCPUKernelFp16::Execute(int task_id) {
  auto dy = reinterpret_cast<float16_t *>(in_tensors_[0]->data());
  auto dx1 = reinterpret_cast<float16_t *>(out_tensors_[0]->data());
  auto dx2 = reinterpret_cast<float16_t *>(out_tensors_[1]->data());
  CHECK_NULL_RETURN(dy);
  CHECK_NULL_RETURN(dx1);
  CHECK_NULL_RETURN(dx2);
  size_t dy_size = in_tensors_.at(0)->ElementsNum();
  size_t dx1_size = out_tensors_.at(0)->ElementsNum();
  size_t dx2_size = out_tensors_.at(1)->ElementsNum();
  (this->*arithmetic_grad_)(dy, dy_size, dx1, dx1_size, dx2, dx2_size);
  return RET_OK;
}

int ArithmeticGradRunFp16(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto Arithmetic_kernel = reinterpret_cast<ArithmeticGradCPUKernelFp16 *>(cdata);
  CHECK_NULL_RETURN(Arithmetic_kernel);
  auto error_code = Arithmetic_kernel->Execute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "ArithmeticGradRunFp16 error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ArithmeticGradCPUKernelFp16::Run() {
  int error_code = ParallelLaunch(this->ms_context_, ArithmeticGradRunFp16, this, 1);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Arithmetic Grad function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_MaximumGrad, LiteKernelCreator<ArithmeticGradCPUKernelFp16>)
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_MinimumGrad, LiteKernelCreator<ArithmeticGradCPUKernelFp16>)
}  // namespace mindspore::kernel
