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
#include <algorithm>
#include "nnacl/int8/sigmoid_int8.h"
#include "mindspore/lite/nnacl/int8/quantize.h"
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
void CalculateTableList(int8_t *table, const float input_scale, const int32_t input_zp) {
  int32_t min_value = std::numeric_limits<int8_t>::min();
  int32_t max_value = std::numeric_limits<int8_t>::max();
  const float output_scale = 1.0f / 256;
  const int32_t output_zp = -128;

  for (int i = min_value; i < max_value; ++i) {
    const float real_input_value = input_scale * (i - input_zp);
    const float sigmoid_value = 1.0f / (1.0f + std::exp(-real_input_value));
    const int32_t quantized = std::round(sigmoid_value / output_scale) + output_zp;
    int8_t out_value = static_cast<int8_t>(std::max(std::min(quantized, max_value), min_value));
    uint8_t index = static_cast<uint8_t>(i);
    table[index] = out_value;
  }
}

int SigmoidInt8CPUKernel::Init() {
  lite::Tensor *input = in_tensors_.at(0);
  lite::Tensor *output = out_tensors_.at(0);
  const float input_scale = input->quant_params().front().scale;
  const int32_t input_zp = input->quant_params().front().zeroPoint;
  const float output_scale = output->quant_params().front().scale;
  const int32_t output_zp = output->quant_params().front().zeroPoint;
  if (output_scale != (1.0f / 256) || output_zp != -128) {
    MS_LOG(ERROR) << "Output scale is : " << output_scale << ", should be 1/256. Output zp is : " << output_zp
                  << ", should be -128.";
    return RET_ERROR;
  }
  CalculateTableList(table_list_, input_scale, input_zp);
  return RET_OK;
}

int SigmoidInt8CPUKernel::ReSize() { return RET_OK; }

int SigmoidInt8CPUKernel::DoActivation(int task_id) {
  auto input_addr = reinterpret_cast<int8_t *>(in_tensors_.at(0)->MutableData());
  MS_ASSERT(input_addr);
  auto output_addr = reinterpret_cast<int8_t *>(out_tensors_.at(0)->MutableData());
  MS_ASSERT(output_addr);
  auto length = in_tensors_.at(0)->ElementsNum();
  int stride = UP_DIV(length, op_parameter_->thread_num_);
  int count = MSMIN(stride, length - stride * task_id);

  SigmoidInt8(input_addr + stride * task_id, count, output_addr + stride * task_id, table_list_);
  return RET_OK;
}

int SigmoidInt8Run(void *cdata, int task_id) {
  auto activation_kernel = reinterpret_cast<SigmoidInt8CPUKernel *>(cdata);
  auto error_code = activation_kernel->DoActivation(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "SigmoidInt8Run error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int SigmoidInt8CPUKernel::Run() {
  int error_code = ParallelLaunch(this->context_->thread_pool_, SigmoidInt8Run, this, op_parameter_->thread_num_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "SigmoidInt8Run function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
