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
#include "src/litert/kernel/cpu/int8/unsqueeze_int8.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_Unsqueeze;

namespace mindspore::kernel {
int Unsqueezeint8CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(in_tensors_[0]);
  CHECK_NULL_RETURN(out_tensors_[0]);
  if (in_tensors_[0]->data_type() != mindspore::kNumberTypeInt8 ||
      out_tensors_[0]->data_type() != mindspore::kNumberTypeInt8) {
    MS_LOG(ERROR) << "Datatype error, input0 data_type is " << in_tensors_[0]->data_type() << ", output data_type is "
                  << out_tensors_[0]->data_type();
    return RET_ERROR;
  }
  auto *input_tensor = in_tensors_.at(0);
  auto quant_params = input_tensor->quant_params();
  MS_CHECK_TRUE_RET(quant_params.size() == 1, RET_ERROR);
  param_->quant_arg.in_quant_args_.scale_ = static_cast<float>(quant_params.front().scale);
  param_->quant_arg.in_quant_args_.zp_ = quant_params.front().zeroPoint;

  auto out_quant_args = input_tensor->quant_params();
  MS_CHECK_TRUE_RET(!quant_params.empty(), RET_ERROR);
  param_->quant_arg.out_quant_args_.scale_ = static_cast<float>(out_quant_args.front().scale);
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
  MS_CHECK_TRUE_MSG(thread_sz_count_ != 0, RET_ERROR, "div zero, multi-thread division failed.");
  thread_sz_stride_ = UP_DIV(data_size_, thread_sz_count_);
  return RET_OK;
}

int Unsqueezeint8CPUKernel::DoUnsqueeze(int task_id) {
  size_t size = MSMIN(thread_sz_stride_, data_size_ - task_id * thread_sz_stride_);
  if (size == 0) {
    return RET_OK;
  }

  auto input_ptr = reinterpret_cast<int8_t *>(in_tensors_.front()->data());
  CHECK_NULL_RETURN(input_ptr);
  auto output_ptr = reinterpret_cast<int8_t *>(out_tensors_.front()->data());
  CHECK_NULL_RETURN(output_ptr);
  size_t data_size = out_tensors_.front()->Size();

  int ret = Int8Unsqueeze(input_ptr, output_ptr, param_, data_size, task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "UnsqueezeRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int UnsqueezeIn8Run(void *cdata, int task_id, float, float) {
  CHECK_NULL_RETURN(cdata);
  auto g_kernel = reinterpret_cast<Unsqueezeint8CPUKernel *>(cdata);
  auto ret = g_kernel->DoUnsqueeze(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "UnsqueezeRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int Unsqueezeint8CPUKernel::Run() {
  in_ptr_ = reinterpret_cast<float *>(in_tensors_.at(0)->data());
  CHECK_NULL_RETURN(in_ptr_);
  out_ptr_ = reinterpret_cast<float *>(out_tensors_.at(0)->data());
  CHECK_NULL_RETURN(out_ptr_);
  auto ret = ParallelLaunch(this->ms_context_, UnsqueezeIn8Run, this, thread_sz_count_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "UnsqueezeRun error error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_Unsqueeze, LiteKernelCreator<Unsqueezeint8CPUKernel>)
}  // namespace mindspore::kernel
