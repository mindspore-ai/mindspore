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

#include "src/runtime/kernel/arm/int8/slice_int8.h"
#include <limits>
#include "src/runtime/kernel/arm/nnacl/slice_parameter.h"
#include "src/runtime/kernel/arm/nnacl/int8/slice_int8.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {

int SliceInt8CPUKernel::Init() {
  auto ret = SliceBaseCPUKernel::Init();
  if (ret != RET_OK) {
    return ret;
  }

  auto input = in_tensors_.at(0);
  auto output = out_tensors_.at(0);
  MS_ASSERT(input);
  MS_ASSERT(output);

  auto in_quant_args = input->GetQuantParams();
  param_->quant_arg_.in_args_.scale_ = in_quant_args.front().scale;
  param_->quant_arg_.in_args_.zp_ = in_quant_args.front().zeroPoint;

  auto out_quant_args = output->GetQuantParams();
  param_->quant_arg_.out_args_.scale_ = out_quant_args.front().scale;
  param_->quant_arg_.out_args_.zp_ = out_quant_args.front().zeroPoint;

  param_->quant_arg_.output_activation_max_ = std::numeric_limits<int8_t>::max();
  param_->quant_arg_.output_activation_min_ = std::numeric_limits<int8_t>::min();
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int SliceInt8CPUKernel::ReSize() { return SliceBaseCPUKernel::ReSize(); }

int SliceInt8CPUKernel::DoSlice(int task_id) {
  const int8_t *input_data = reinterpret_cast<const int8_t *>(in_tensors_[0]->Data());
  int8_t *output_data = reinterpret_cast<int8_t *>(out_tensors_[0]->Data());

  param_->thread_id_ = task_id;
  auto ret = SliceInt8(input_data, output_data, param_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SliceInt8 error ,task_id[" << task_id << "] error_code[" << ret << "]";
  }
  return ret;
}

int SliceInt8Run(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto slice_kernel = reinterpret_cast<SliceInt8CPUKernel *>(cdata);
  auto ret = slice_kernel->DoSlice(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DoSlice error, task_id[" << task_id << "] error_code[" << ret << "]";
  }
  return ret;
}

int SliceInt8CPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare failed.";
    return ret;
  }

  const int8_t *input_data = reinterpret_cast<const int8_t *>(in_tensors_[0]->Data());
  int8_t *output_data = reinterpret_cast<int8_t *>(out_tensors_[0]->Data());

  if (param_->size_[1] < param_->op_parameter_.thread_num_) {
    ret = SliceInt8NoParallel(input_data, output_data, param_);
  } else {
    ret = LiteBackendParallelLaunch(SliceInt8Run, this, op_parameter_->thread_num_);
  }

  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SliceInt8Run error, error_code[" << ret << "]";
  }
  return ret;
}
}  // namespace mindspore::kernel
