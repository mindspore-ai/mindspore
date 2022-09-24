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

#include "src/litert/kernel/cpu/int8/hswish_int8.h"
#include <limits>
#include "nnacl/int8/hswish_int8.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::ActivationType_HSWISH;

namespace mindspore::kernel {
int HswishInt8CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C1NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), C1NUM);
  CHECK_NULL_RETURN(in_tensors_[0]);
  CHECK_NULL_RETURN(out_tensors_[0]);
  if (in_tensors_[0]->data_type() != mindspore::kNumberTypeInt8 ||
      out_tensors_[0]->data_type() != mindspore::kNumberTypeInt8) {
    MS_LOG(ERROR) << "Datatype error, input0 data_type is " << in_tensors_[0]->data_type() << ", output data_type is "
                  << out_tensors_[0]->data_type();
    return RET_ERROR;
  }
  lite::Tensor *input = in_tensors_.at(0);
  lite::Tensor *output = out_tensors_.at(0);
  const auto &in_quant_args = input->quant_params();
  const auto &out_quant_args = output->quant_params();
  MS_CHECK_TRUE_MSG(!in_quant_args.empty(), RET_ERROR, "Input quant param cannot be empty.");
  MS_CHECK_TRUE_MSG(!out_quant_args.empty(), RET_ERROR, "Output quant param cannot be empty.");

  quant_arg_.input_scale = in_quant_args.front().scale;
  quant_arg_.input_zp = in_quant_args.front().zeroPoint;
  quant_arg_.output_scale = out_quant_args.front().scale;
  quant_arg_.output_zp = out_quant_args.front().zeroPoint;

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

void HswishInt8CPUKernel::MultiplierInt32ToInt16(int32_t input, int16_t *output) const {
  MS_ASSERT(input >= 0);
  if (input >= std::numeric_limits<int32_t>::max() - (1 << 15)) {
    *output = std::numeric_limits<int16_t>::max();
    return;
  }
  *output = (input + (1 << 15)) >> 16;
}

int HswishInt8CPUKernel::ReSize() { return RET_OK; }

int HswishInt8CPUKernel::DoActivation(int task_id) {
  auto input_addr = reinterpret_cast<int8_t *>(in_tensors_.at(0)->MutableData());
  auto output_addr = reinterpret_cast<int8_t *>(out_tensors_.at(0)->MutableData());
  auto length = in_tensors_.at(0)->ElementsNum();
  MS_CHECK_GT(length, 0, RET_ERROR);
  int stride = UP_DIV(length, thread_count_);
  int count = MSMIN(stride, length - stride * task_id);

  auto ret = HSwishInt8(input_addr + stride * task_id, count, output_addr + stride * task_id, &quant_arg_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DoActivation hswish int8 task id " << task_id << " failed.";
    return ret;
  }
  return RET_OK;
}

int HswishInt8Run(void *cdata, int task_id, float, float) {
  auto activation_kernel = reinterpret_cast<HswishInt8CPUKernel *>(cdata);
  auto error_code = activation_kernel->DoActivation(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "HswishInt8Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int HswishInt8CPUKernel::Run() {
  int error_code = ParallelLaunch(this->ms_context_, HswishInt8Run, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "HswishInt8Run function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
