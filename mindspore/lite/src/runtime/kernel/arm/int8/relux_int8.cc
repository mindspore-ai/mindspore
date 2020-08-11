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

#include "src/runtime/kernel/arm/int8/relux_int8.h"
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
int ReluXInt8CPUKernel::Init() {
  lite::tensor::Tensor *input = in_tensors_.at(0);
  lite::tensor::Tensor *output = out_tensors_.at(0);
  MS_ASSERT(input);
  MS_ASSERT(output);

  quant_arg_.input_arg.scale_ = input->GetQuantParams().front().scale;
  quant_arg_.input_arg.zp_ = input->GetQuantParams().front().zeroPoint;
  quant_arg_.output_arg.scale_ = output->GetQuantParams().front().scale;
  quant_arg_.output_arg.zp_ = output->GetQuantParams().front().zeroPoint;

  const double multiplier = quant_arg_.input_arg.scale_ / quant_arg_.output_arg.scale_;
  QuantizeRoundParameter(multiplier, &quant_arg_.input_multiplier_, &quant_arg_.left_shift_, &quant_arg_.right_shift_);

  return RET_OK;
}

int ReluXInt8CPUKernel::ReSize() { return RET_OK; }

int ReluXInt8CPUKernel::DoActivation(int task_id) {
  auto input_addr = reinterpret_cast<int8_t *>(in_tensors_.at(0)->Data());
  auto output_addr = reinterpret_cast<int8_t *>(out_tensors_.at(0)->Data());
  auto length = in_tensors_.at(0)->ElementsNum();

  int stride = UP_DIV(length, op_parameter_->thread_num_);
  int count = MSMIN(stride, length - stride * task_id);

  ReluXInt8(input_addr + stride * task_id, count, output_addr + stride * task_id, &quant_arg_);
  return RET_OK;
}

int ReluXInt8Run(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto activation_kernel = reinterpret_cast<ReluXInt8CPUKernel *>(cdata);
  auto error_code = activation_kernel->DoActivation(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "ReluXInt8Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ReluXInt8CPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << ret;
    return ret;
  }
  int error_code = LiteBackendParallelLaunch(ReluXInt8Run, this, op_parameter_->thread_num_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "ReluXInt8Run function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
