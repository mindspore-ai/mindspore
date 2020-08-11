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

#include "src/runtime/kernel/arm/int8/sigmoid_int8.h"
#include <limits>
#include "src/runtime/kernel/arm/nnacl/int8/sigmoid_int8.h"
#include "src/runtime/kernel/arm/nnacl/quantization/quantize.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::ActivationType_SIGMOID;

namespace mindspore::kernel {
int SigmoidInt8CPUKernel::Init() {
  lite::tensor::Tensor *input = in_tensors_.at(0);
  lite::tensor::Tensor *output = out_tensors_.at(0);
  MS_ASSERT(input);
  MS_ASSERT(output);

  quant_arg_.input_scale = input->GetQuantParams().front().scale;
  quant_arg_.input_zp = input->GetQuantParams().front().zeroPoint;
  quant_arg_.output_scale = output->GetQuantParams().front().scale;
  quant_arg_.output_zp = output->GetQuantParams().front().zeroPoint;

  const float output_multiplier = (1.0f / 128.0f) * quant_arg_.input_scale / quant_arg_.output_scale;

  int32_t output_multiplier_fixedpoint;
  QuantizeMultiplier(output_multiplier, &output_multiplier_fixedpoint, &quant_arg_.output_multiplier_exponent);
  MS_ASSERT(quant_arg_.output_multiplier_exponent <= 0);
  MultiplierInt32ToInt16(output_multiplier_fixedpoint, &quant_arg_.output_multiplier_fixedpoint_int16);

  const float relu6_multiplier = (1.0f / 128.0f) * quant_arg_.input_scale / (3.0f / 32768.0f);
  int32_t relu6_multiplier_fixedpoint;
  QuantizeMultiplier(relu6_multiplier, &relu6_multiplier_fixedpoint, &quant_arg_.relu6_multiplier_exponent);
  MultiplierInt32ToInt16(relu6_multiplier_fixedpoint, &quant_arg_.relu6_multiplier_fixedpoint_int16);

  return RET_OK;
}

void SigmoidInt8CPUKernel::MultiplierInt32ToInt16(int32_t input, int16_t *output) {
  MS_ASSERT(input >= 0);
  if (input >= std::numeric_limits<int32_t>::max() - (1 << 15)) {
    *output = std::numeric_limits<int16_t>::max();
    return;
  }
  *output = (input + (1 << 15)) >> 16;
}

int SigmoidInt8CPUKernel::ReSize() { return RET_OK; }

int SigmoidInt8CPUKernel::DoActivation(int task_id) {
  auto input_addr = reinterpret_cast<int8_t *>(in_tensors_.at(0)->Data());
  auto output_addr = reinterpret_cast<int8_t *>(out_tensors_.at(0)->Data());
  auto length = in_tensors_.at(0)->ElementsNum();

  int stride = UP_DIV(length, op_parameter_->thread_num_);
  int count = MSMIN(stride, length - stride * task_id);

  SigmoidInt8(input_addr + stride * task_id, count, output_addr + stride * task_id, &quant_arg_);
  return RET_OK;
}

int SigmoidInt8Run(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto activation_kernel = reinterpret_cast<SigmoidInt8CPUKernel *>(cdata);
  auto error_code = activation_kernel->DoActivation(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "SigmoidInt8Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int SigmoidInt8CPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << ret;
    return ret;
  }
  int error_code = LiteBackendParallelLaunch(SigmoidInt8Run, this, op_parameter_->thread_num_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "SigmoidInt8Run function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
