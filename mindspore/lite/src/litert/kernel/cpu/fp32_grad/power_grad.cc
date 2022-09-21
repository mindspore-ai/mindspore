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

#include "src/litert/kernel/cpu/fp32_grad/power_grad.h"
#include "plugin/device/cpu/kernel/nnacl/op_base.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/fp32/arithmetic_fp32.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_PowerGrad;

namespace mindspore::kernel {
int PowerGradCPUKernel::Prepare() {
  if (in_tensors_.size() != 2) {
    MS_LOG(ERROR) << "Power Grad Filter should have 2 inputs";
    return RET_ERROR;
  }
  CHECK_NULL_RETURN(in_tensors_.at(0));
  CHECK_NULL_RETURN(in_tensors_.at(1));
  if (out_tensors_.size() != 1) {
    MS_LOG(ERROR) << "Power Grad Filter should have one output";
    return RET_ERROR;
  }
  CHECK_NULL_RETURN(out_tensors_.at(0));
  return RET_OK;
}

int PowerGradCPUKernel::ReSize() { return RET_OK; }

int PowerGradCPUKernel::DoExecute(int task_id) {
  auto dy_addr = reinterpret_cast<float *>(in_tensors_.at(0)->MutableData());
  auto x_addr = reinterpret_cast<float *>(in_tensors_.at(1)->MutableData());
  auto dx_addr = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());
  CHECK_NULL_RETURN(dy_addr);
  CHECK_NULL_RETURN(x_addr);
  CHECK_NULL_RETURN(dx_addr);
  int length = in_tensors_.at(0)->ElementsNum();

  int stride = UP_DIV(length, thread_count_);
  int count = MSMIN(stride, length - stride * task_id);
  count = (count < 0) ? 0 : count;
  int start = stride * task_id;
  int end = start + count;

  float exp = power_ - 1;
  Power(&(x_addr[start]), &exp, &(dx_addr[start]), count, scale_, shift_, true);
  ElementMul(&(dx_addr[start]), &(dy_addr[start]), &(dx_addr[start]), count);
  float scale = scale_ * power_;
  for (int i = start; i < end; i++) {
    dx_addr[i] *= scale;
  }

  return RET_OK;
}

int PowerGradRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  auto power_kernel = reinterpret_cast<PowerGradCPUKernel *>(cdata);
  CHECK_NULL_RETURN(power_kernel);
  auto error_code = power_kernel->DoExecute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "power grad error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int PowerGradCPUKernel::Run() {
  int error_code = ParallelLaunch(this->ms_context_, PowerGradRun, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "power grad function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuPowerGradFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                  const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                                  const lite::InnerContext *ctx, const kernel::KernelKey &desc) {
  MS_CHECK_TRUE_MSG(opParameter != nullptr, nullptr, "Op parameter is nullptr.");
  MS_ASSERT(desc.type == schema::PrimitiveType_PowerGrad);
  auto *kernel = new (std::nothrow) PowerGradCPUKernel(opParameter, inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new PowerGradCPUKernel fail!";
    free(opParameter);
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_PowerGrad, CpuPowerGradFp32KernelCreator)
}  // namespace mindspore::kernel
