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
#include "src/kernel_registry.h"
#include "nnacl/int8/slice_int8.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_SliceFusion;

namespace mindspore::kernel {

int SliceInt8CPUKernel::Init() {
  auto input = in_tensors_.at(0);
  auto output = out_tensors_.at(0);
  MS_ASSERT(input);
  MS_ASSERT(output);

  auto in_quant_args = input->quant_params();
  param_->quant_arg_.in_args_.scale_ = in_quant_args.front().scale;
  param_->quant_arg_.in_args_.zp_ = in_quant_args.front().zeroPoint;

  auto out_quant_args = output->quant_params();
  param_->quant_arg_.out_args_.scale_ = out_quant_args.front().scale;
  param_->quant_arg_.out_args_.zp_ = out_quant_args.front().zeroPoint;

  param_->quant_arg_.output_activation_max_ = std::numeric_limits<int8_t>::max();
  param_->quant_arg_.output_activation_min_ = std::numeric_limits<int8_t>::min();
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int SliceInt8CPUKernel::DoSlice(int task_id) {
  const int8_t *input_data = reinterpret_cast<const int8_t *>(in_tensors_.at(0)->MutableData());
  MS_ASSERT(input_data);
  int8_t *output_data = reinterpret_cast<int8_t *>(out_tensors_.at(0)->MutableData());
  MS_ASSERT(output_data);

  auto ret = SliceInt8(input_data, output_data, param_, task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SliceInt8 error ,task_id[" << task_id << "] error_code[" << ret << "]";
  }
  return ret;
}

int SliceInt8Run(void *cdata, int task_id) {
  auto slice_kernel = reinterpret_cast<SliceInt8CPUKernel *>(cdata);
  auto ret = slice_kernel->DoSlice(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DoSlice error, task_id[" << task_id << "] error_code[" << ret << "]";
  }
  return ret;
}

int SliceInt8CPUKernel::Run() {
  const int8_t *input_data = reinterpret_cast<const int8_t *>(in_tensors_.at(0)->MutableData());
  MS_ASSERT(input_data);
  int8_t *output_data = reinterpret_cast<int8_t *>(out_tensors_.at(0)->MutableData());
  MS_ASSERT(output_data);
  mindspore::lite::STATUS ret = RET_ERROR;
  if (param_->size_[1] < param_->op_parameter_.thread_num_) {
    ret = SliceInt8NoParallel(input_data, output_data, param_);
  } else {
    ret = ParallelLaunch(this->context_->thread_pool_, SliceInt8Run, this, op_parameter_->thread_num_);
  }

  if (ret != RET_OK) {
    MS_LOG(ERROR) << "SliceInt8Run error, error_code[" << ret << "]";
  }
  return ret;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_SliceFusion, LiteKernelCreator<SliceInt8CPUKernel>)
}  // namespace mindspore::kernel
