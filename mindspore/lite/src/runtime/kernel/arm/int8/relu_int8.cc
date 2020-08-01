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

#include "src/runtime/kernel/arm/int8/relu_int8.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::ActivationType_RELU;

namespace mindspore::kernel {
int ReluInt8CPUKernel::Init() {
  lite::tensor::Tensor *input = inputs_.at(0);
  lite::tensor::Tensor *output = outputs_.at(0);
  MS_ASSERT(input);
  MS_ASSERT(output);

  quant_arg_.input_arg.scale_ = input->GetQuantParams().front().scale;
  quant_arg_.input_arg.zp_ = input->GetQuantParams().front().zeroPoint;
  quant_arg_.output_arg.scale_ = output->GetQuantParams().front().scale;
  quant_arg_.output_arg.zp_ = output->GetQuantParams().front().zeroPoint;

  const double multiplier = quant_arg_.input_arg.scale_ / quant_arg_.output_arg.scale_;
  QuantizeMultiplierSmallerThanOne(multiplier, &quant_arg_.input_multiplier_, &quant_arg_.input_shift_);

  int left_shift = -quant_arg_.input_shift_ > 0 ? -quant_arg_.input_shift_ : 0;
  quant_arg_.right_shift_ = -quant_arg_.input_shift_ > 0 ? 0 : quant_arg_.input_shift_;
  quant_arg_.left_shift_result_ = (1 << left_shift);

  return RET_OK;
}

int ReluInt8CPUKernel::ReSize() { return RET_OK; }

int ReluInt8CPUKernel::DoActivation(int task_id) {
  auto input_addr = reinterpret_cast<int8_t *>(inputs_.at(0)->Data());
  auto output_addr = reinterpret_cast<int8_t *>(outputs_.at(0)->Data());
  auto length = inputs_.at(0)->ElementsNum();

  int stride = UP_DIV(length, thread_count_);
  int count = MSMIN(stride, length - stride * task_id);

  ReluInt8(input_addr + stride * task_id, count, output_addr + stride * task_id, &quant_arg_);
  return RET_OK;
}

int ReluInt8Run(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto activation_kernel = reinterpret_cast<ReluInt8CPUKernel *>(cdata);
  auto error_code = activation_kernel->DoActivation(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "ReluInt8Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ReluInt8CPUKernel::Run() {
  int error_code = LiteBackendParallelLaunch(ReluInt8Run, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "ReluInt8Run function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
