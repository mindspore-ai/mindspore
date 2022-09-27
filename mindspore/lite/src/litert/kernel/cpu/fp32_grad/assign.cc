
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

#include "src/litert/kernel/cpu/fp32_grad/assign.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Assign;

namespace mindspore::kernel {
int AssignCPUKernel::ReSize() { return RET_OK; }

int AssignCPUKernel::DoExecute(int task_id) {
  auto x = reinterpret_cast<float *>(in_tensors_.at(0)->data());
  CHECK_NULL_RETURN(x);
  auto y = reinterpret_cast<float *>(in_tensors_.at(1)->data());
  CHECK_NULL_RETURN(y);
  int length = in_tensors_.at(0)->ElementsNum();
  int stride = UP_DIV(length, thread_count_);
  int count = MSMIN(stride, length - stride * task_id);

  int start = stride * task_id;

  if (count > 0) {
    memcpy(&(x[start]), &(y[start]), static_cast<size_t>(count) * sizeof(float));
  }
  return RET_OK;
}

int AssignRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  CHECK_NULL_RETURN(cdata);
  auto Assign_kernel = reinterpret_cast<AssignCPUKernel *>(cdata);
  auto error_code = Assign_kernel->DoExecute(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "assign run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int AssignCPUKernel::Run() {
  int error_code = ParallelLaunch(this->ms_context_, AssignRun, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Assign function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int AssignCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), 2);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(in_tensors_.at(0));
  CHECK_NULL_RETURN(in_tensors_.at(1));
  CHECK_NULL_RETURN(out_tensors_.at(0));
  return RET_OK;
}

kernel::LiteKernel *CpuAssignFp32KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                               const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                               const lite::InnerContext *ctx, const kernel::KernelKey &desc) {
  MS_ASSERT(desc.type == schema::PrimitiveType_Assign);
  auto *kernel = new (std::nothrow) AssignCPUKernel(opParameter, inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new AssignCPUKernel fail!";
    free(opParameter);
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Assign, CpuAssignFp32KernelCreator)
}  // namespace mindspore::kernel
