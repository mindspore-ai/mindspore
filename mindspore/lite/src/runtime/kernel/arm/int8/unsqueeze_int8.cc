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

#include "nnacl/int8/unsqueeze_int8.h"
#include "src/runtime/kernel/arm/int8/unsqueeze_int8.h"
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
int Unsqueezeint8CPUKernel::Init() {
  auto *input_tensor = in_tensors_.at(0);
  auto quant_args = input_tensor->quant_params();
  MS_ASSERT(quant_args.size() == 1);
  param_->quant_arg.in_quant_args_.scale_ = quant_args.front().scale;
  param_->quant_arg.in_quant_args_.zp_ = quant_args.front().zeroPoint;

  auto out_quant_args = input_tensor->quant_params();
  param_->quant_arg.out_quant_args_.scale_ = out_quant_args.front().scale;
  param_->quant_arg.out_quant_args_.zp_ = out_quant_args.front().zeroPoint;
  param_->thread_count_ = thread_count_;
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int Unsqueezeint8CPUKernel::ReSize() {
  data_size_ = in_tensors_.at(0)->ElementsNum();
  thread_sz_count_ = MSMIN(thread_count_, data_size_);
  thread_sz_stride_ = UP_DIV(data_size_, thread_sz_count_);
  return RET_OK;
}

int Unsqueezeint8CPUKernel::DoUnsqueeze(int task_id) {
  size_t size = MSMIN(thread_sz_stride_, data_size_ - task_id * thread_sz_stride_);
  if (size == 0) {
    return RET_OK;
  }

  auto input_ptr = reinterpret_cast<int8_t *>(in_tensors_.front()->MutableData());
  MS_ASSERT(input_ptr);
  auto output_ptr = reinterpret_cast<int8_t *>(out_tensors_.front()->MutableData());
  MS_ASSERT(output_ptr);
  size_t data_size = out_tensors_.front()->Size();

  int ret = Int8Unsqueeze(input_ptr, output_ptr, param_, data_size, task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "UnsqueezeRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int UnsqueezeIn8Run(void *cdata, int task_id) {
  auto g_kernel = reinterpret_cast<Unsqueezeint8CPUKernel *>(cdata);
  auto ret = g_kernel->DoUnsqueeze(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "UnsqueezeRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int Unsqueezeint8CPUKernel::Run() {
  in_ptr_ = reinterpret_cast<float *>(in_tensors_.at(0)->MutableData());
  out_ptr_ = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());
  auto ret = ParallelLaunch(this->context_->thread_pool_, UnsqueezeIn8Run, this, thread_sz_count_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "UnsqueezeRun error error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Unsqueeze, LiteKernelCreator<Unsqueezeint8CPUKernel>)
}  // namespace mindspore::kernel
