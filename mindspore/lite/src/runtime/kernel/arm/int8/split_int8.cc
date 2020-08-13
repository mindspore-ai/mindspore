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

#include "src/runtime/kernel/arm/int8/split_int8.h"
#include <limits>
#include "src/runtime/kernel/arm/nnacl/split_parameter.h"
#include "src/runtime/kernel/arm/nnacl/int8/split_int8.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {

int SplitInt8CPUKernel::Init() {
  auto ret = SplitBaseCPUKernel::Init();
  if (ret != RET_OK) {
    return ret;
  }
  auto in_tensor = in_tensors_.at(kInputIndex);

  auto in_quant_args = in_tensor->GetQuantParams();
  param->quant_arg_.in_args_.scale_ = in_quant_args.front().scale;
  param->quant_arg_.in_args_.zp_ = in_quant_args.front().zeroPoint;
  MS_ASSERT(param->num_split_ == outputs_.size());
  for (int i = 0; i < param->num_split_; i++) {
    auto *out_tensor = out_tensors_.at(i);
    auto out_quant_args = out_tensor->GetQuantParams();
    param->quant_arg_.out_args_[i].scale_ = out_quant_args.front().scale;
    param->quant_arg_.out_args_[i].zp_ = out_quant_args.front().zeroPoint;
  }

  param->quant_arg_.output_activation_max_ = std::numeric_limits<int8_t>::max();
  param->quant_arg_.output_activation_min_ = std::numeric_limits<int8_t>::min();
  if (!InferShapeDone()) {
    return RET_OK;
  }

  return ReSize();
}

int SplitInt8CPUKernel::ReSize() { return SplitBaseCPUKernel::ReSize(); }

int SplitInt8CPUKernel::Split(int task_id) {
  int num_unit_thread = MSMIN(thread_n_stride_, num_unit_ - task_id * thread_n_stride_);
  if (num_unit_thread <= 0) {
    return RET_OK;
  }
  int thread_offset = task_id * thread_n_stride_;
  auto ret = Int8DoSplit(input_ptr_, output_ptr_.data(), in_tensors_.front()->shape().data(), thread_offset,
                         num_unit_thread, param);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Split error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int SplitInt8Run(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto g_kernel = reinterpret_cast<SplitInt8CPUKernel *>(cdata);
  auto ret = g_kernel->Split(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SplitRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int SplitInt8CPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare failed.";
    return ret;
  }
  auto in_tensor = in_tensors_.at(kInputIndex);
  input_ptr_ = reinterpret_cast<int8_t *>(in_tensor->Data());
  MS_ASSERT(param->num_split_ == outputs_.size());
  for (int i = 0; i < param->num_split_; i++) {
    output_ptr_.push_back(reinterpret_cast<int8_t *>(out_tensors_.at(i)->Data()));
  }

  ret = LiteBackendParallelLaunch(SplitInt8Run, this, thread_n_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Scale error error_code[" << ret << "]";
    return RET_ERROR;
  }

  return RET_OK;
}
}  // namespace mindspore::kernel
