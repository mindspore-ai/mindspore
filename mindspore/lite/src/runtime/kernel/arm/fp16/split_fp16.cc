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
#include "src/runtime/kernel/arm/fp16/split_fp16.h"
#include "src/runtime/kernel/arm/fp16/common_fp16.h"
#include "src/runtime/kernel/arm/base/split_base.h"
#include "nnacl/fp16/split_fp16.h"
#include "nnacl/split.h"
#include "nnacl/split_parameter.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Split;

namespace mindspore::kernel {
int SplitFp16CPUKernel::Init() {
  auto ret = SplitBaseCPUKernel::Init();
  if (ret != RET_OK) {
    return ret;
  }
  output_ptr_.resize(param->num_split_);
  for (size_t i = 0; i < output_ptr_.size(); i++) {
    output_ptr_.at(i) = nullptr;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }

  return ReSize();
}

int SplitFp16CPUKernel::ReSize() { return SplitBaseCPUKernel::ReSize(); }

int SplitFp16CPUKernel::Split(int task_id) {
  int num_unit_thread = MSMIN(thread_n_stride_, num_unit_ - task_id * thread_n_stride_);
  if (num_unit_thread <= 0) {
    return RET_OK;
  }
  int thread_offset = task_id * thread_n_stride_;
  auto ret = DoSplitFp16(input_ptr_, output_ptr_.data(), in_tensors_.front()->shape().data(), thread_offset,
                         num_unit_thread, param);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Split error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

static int SplitFp16Run(void *cdata, int task_id) {
  auto g_kernel = reinterpret_cast<SplitFp16CPUKernel *>(cdata);
  auto ret = g_kernel->Split(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SplitRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int SplitFp16CPUKernel::Run() {
  auto input_tensor = in_tensors_.at(0);
  input_ptr_ = reinterpret_cast<float16_t *>(input_tensor->data_c());

  for (int i = 0; i < param->num_split_; i++) {
    auto output_tensor = out_tensors_.at(i);
    output_ptr_.at(i) = reinterpret_cast<float16_t *>(output_tensor->data_c());
  }

  auto ret = ParallelLaunch(this->context_->thread_pool_, SplitFp16Run, this, thread_n_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "split error error_code[" << ret << "]";
  }
  return ret;
}
REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Split, LiteKernelCreator<SplitFp16CPUKernel>)
}  // namespace mindspore::kernel
