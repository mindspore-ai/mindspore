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

int ArithmeticGradCPUKernelFp16::Init() { return RET_OK; }

void ArithmeticGradCPUKernelFp16::ArithmeticGradMaximum(float16_t *dy, int dy_size, float16_t *dx1, int dx1_size,
                                                        float16_t *dx2, int dx2_size) {
  auto x1 = reinterpret_cast<float16_t *>(in_tensors_[0]->data_c());
  auto x2 = reinterpret_cast<float16_t *>(in_tensors_[1]->data_c());
  dy = reinterpret_cast<float16_t *>(in_tensors_[2]->data_c());

  MaximumByAxesFp16(x1, x2, dy, arithmeticParameter_->in_shape0_, arithmeticParameter_->in_shape1_,
                    arithmeticParameter_->out_shape_, dx1, dx2, arithmeticParameter_->ndim_);
}

void ArithmeticGradCPUKernelFp16::ArithmeticGradMinimum(float16_t *dy, int dy_size, float16_t *dx1, int dx1_size,
                                                        float16_t *dx2, int dx2_size) {
  auto x1 = reinterpret_cast<float16_t *>(in_tensors_[0]->data_c());
  auto x2 = reinterpret_cast<float16_t *>(in_tensors_[1]->data_c());
  dy = reinterpret_cast<float16_t *>(in_tensors_[2]->data_c());

  MinimumByAxesFp16(x1, x2, dy, arithmeticParameter_->in_shape0_, arithmeticParameter_->in_shape1_,
                    arithmeticParameter_->out_shape_, dx1, dx2, arithmeticParameter_->ndim_);
}

int ArithmeticGradCPUKernelFp16::ReSize() { return RET_OK; }

int ArithmeticGradCPUKernelFp16::Execute(int task_id) {
  auto dy = reinterpret_cast<float16_t *>(in_tensors_[0]->data_c());
  auto dx1 = reinterpret_cast<float16_t *>(out_tensors_[0]->data_c());
  auto dx2 = reinterpret_cast<float16_t *>(out_tensors_[1]->data_c());

  size_t dy_size = in_tensors_.at(0)->ElementsNum();
  size_t dx1_size = out_tensors_.at(0)->ElementsNum();
  size_t dx2_size = out_tensors_.at(1)->ElementsNum();
  (this->*arithmetic_grad_)(dy, dy_size, dx1, dx1_size, dx2, dx2_size);
  return RET_OK;
}

int ArithmeticGradRunFp16(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  MS_ASSERT(cdata != nullptr);
  auto Arithmetic_kernel = reinterpret_cast<ArithmeticGradCPUKernelFp16 *>(cdata);
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
