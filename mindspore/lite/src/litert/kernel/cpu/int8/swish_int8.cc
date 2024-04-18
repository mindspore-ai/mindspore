/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/cpu/int8/swish_int8.h"
#include <limits>
#include <algorithm>
#include "nnacl/int8/quantize.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::ActivationType_SIGMOID;

namespace mindspore::kernel {
//  Calculate the quantization results of 0-255 in advance
void CalculateSwishTableList(int8_t *table, const float input_scale, const int32_t input_zp, const float output_scale,
                             const int32_t output_zp) {
  int32_t min_value = std::numeric_limits<int8_t>::min();
  int32_t max_value = std::numeric_limits<int8_t>::max();
  for (int i = min_value; i < max_value; ++i) {
    const float real_input_value = input_scale * (i - input_zp);
    const float sigmoid_value = 1.0f / (1.0f + std::exp(-real_input_value));
    const int32_t quantized = (std::round(real_input_value * sigmoid_value / output_scale) + output_zp);
    int8_t out_value = static_cast<int8_t>(std::max(std::min(quantized, max_value), min_value));
    uint8_t index = static_cast<uint8_t>(i);
    table[index] = out_value;
  }
}

int SwishInt8CPUKernel::Prepare() {
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
  if (input->quant_params().size() != C1NUM || output->quant_params().size() != C1NUM) {
    MS_LOG(ERROR) << "swish int8 kernel only supports per-tensor quantization, but now input->quant_params().size() is "
                  << input->quant_params().size() << ", output->quant_params().size() is "
                  << output->quant_params().size();
    return RET_ERROR;
  }
  const float input_scale = input->quant_params().front().scale;
  const int32_t input_zp = input->quant_params().front().zeroPoint;
  const float output_scale = output->quant_params().front().scale;
  const int32_t output_zp = output->quant_params().front().zeroPoint;
  CalculateSwishTableList(table_list_, input_scale, input_zp, output_scale, output_zp);
  return RET_OK;
}
}  // namespace mindspore::kernel
