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

#include "src/runtime/kernel/arm/fp32_grad/neg_grad.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/fp32/arithmetic_self_fp32.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_NegGrad;

namespace mindspore::kernel {
namespace {
int NegGradRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  CHECK_NULL_RETURN(cdata);
  auto kernel = reinterpret_cast<NegGradCPUKernel *>(cdata);
  return kernel->DoNegGrad(task_id);
}
}  // namespace

int NegGradCPUKernel::Init() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(in_tensors_.at(0));
  CHECK_NULL_RETURN(out_tensors_.at(0));
  return RET_OK;
}

int NegGradCPUKernel::DoNegGrad(int task_id) {
  auto dy = reinterpret_cast<float *>(in_tensors_.at(0)->MutableData());
  auto dx = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());
  CHECK_NULL_RETURN(dy);
  CHECK_NULL_RETURN(dx);
  int length = in_tensors_.at(0)->ElementsNum();

  int stride = UP_DIV(length, thread_count_);
  int count = MSMIN(stride, length - stride * task_id);
  count = (count < 0) ? 0 : count;
  int start = stride * task_id;

  ElementNegative(dy + start, dx + start, count);
  return RET_OK;
}

int NegGradCPUKernel::ReSize() { return RET_OK; }

int NegGradCPUKernel::Run() {
  auto ret = ParallelLaunch(this->ms_context_, NegGradRun, this, thread_count_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "parallel launch fail!ret: " << ret;
    return ret;
  }

  return RET_OK;
}

kernel::InnerKernel *CpuNegGradFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                 const std::vector<lite::Tensor *> &outputs, OpParameter *param,
                                                 const lite::Context *ctx, const kernel::KernelKey &desc) {
  if (param == nullptr) {
    MS_LOG(ERROR) << "input parameter is nullptr!";
    return nullptr;
  }
  auto *kernel =
    new (std::nothrow) NegGradCPUKernel(param, inputs, outputs, static_cast<const lite::InnerContext *>(ctx));
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new NegGradCPUKernel fail!";
    free(param);
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_NegGrad, CpuNegGradFp32KernelCreator)
}  // namespace mindspore::kernel
