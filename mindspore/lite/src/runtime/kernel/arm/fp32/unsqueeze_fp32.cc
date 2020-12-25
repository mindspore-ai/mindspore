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

#include "src/runtime/kernel/arm/fp32/unsqueeze_fp32.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Unsqueeze;

namespace mindspore::kernel {
int UnsqueezeCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int UnsqueezeCPUKernel::ReSize() {
  data_size_ = in_tensors_.at(0)->ElementsNum();
  thread_sz_count_ = MSMIN(context_->thread_num_, data_size_);
  thread_sz_stride_ = UP_DIV(data_size_, thread_sz_count_);
  return RET_OK;
}

int UnsqueezeCPUKernel::DoUnsqueeze(int task_id) {
  size_t size = MSMIN(thread_sz_stride_, data_size_ - task_id * thread_sz_stride_);
  if (size == 0) {
    return RET_OK;
  }
  size_t offset = task_id * thread_sz_stride_ * sizeof(float);
  MS_ASSERT(in_ptr_);
  MS_ASSERT(out_ptr_);
  int ret = Unsqueeze(in_ptr_ + offset, out_ptr_ + offset, size * sizeof(float));
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "UnsqueezeRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int UnsqueezeRun(void *cdata, int task_id) {
  auto g_kernel = reinterpret_cast<UnsqueezeCPUKernel *>(cdata);
  auto ret = g_kernel->DoUnsqueeze(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "UnsqueezeRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int UnsqueezeCPUKernel::Run() {
  in_ptr_ = reinterpret_cast<int8_t *>(in_tensors_.at(0)->MutableData());
  out_ptr_ = reinterpret_cast<int8_t *>(out_tensors_.at(0)->MutableData());
  auto ret = ParallelLaunch(this->context_->thread_pool_, UnsqueezeRun, this, thread_sz_count_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "UnsqueezeRun error error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}
REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Unsqueeze, LiteKernelCreator<UnsqueezeCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Unsqueeze, LiteKernelCreator<UnsqueezeCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt64, PrimitiveType_Unsqueeze, LiteKernelCreator<UnsqueezeCPUKernel>)
}  // namespace mindspore::kernel
