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

#include "src/runtime/kernel/arm/int8/reshape_int8.h"
#include "src/runtime/kernel/arm/nnacl/int8/reshape_int8.h"
#include "schema/model_generated.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {

int ReshapeInt8CPUKernel::Init() {
  ReshapeBaseCPUKernel::Init();
  auto *input_tensor = inputs_.at(kInputIndex);
  auto in_quant_args = input_tensor->GetQuantParams();
  in_quant_arg_.scale_ = in_quant_args.front().scale;
  in_quant_arg_.zp_ = in_quant_args.front().zeroPoint;

  auto *out_tensor = outputs_.at(kOutputIndex);
  auto out_quant_args = out_tensor->GetQuantParams();
  out_quant_arg_.scale_ = out_quant_args.front().scale;
  out_quant_arg_.zp_ = out_quant_args.front().zeroPoint;
  return RET_OK;
}

int ReshapeInt8CPUKernel::ReSize() { return 0; }

int ReshapeInt8CPUKernel::Run() {
  MS_ASSERT(inputs_.size() == 1);
  MS_ASSERT(outputs_.size() == 1);
  auto input_type = inputs_[kInputIndex]->data_type();
  auto input_num = inputs_[kInputIndex]->ElementsNum();
  auto output_num = outputs_.at(kOutputIndex)->ElementsNum();
  MS_ASSERT(input_num == output_num);
  int8_t *input_ptr = reinterpret_cast<int8_t *>(inputs_.at(kInputIndex)->Data());
  int8_t *output_ptr = reinterpret_cast<int8_t *>(outputs_.at(kOutputIndex)->Data());
  if (input_type == kNumberTypeUInt8) {
    auto *input_tmp = reinterpret_cast<uint8_t *>(inputs_.at(kInputIndex)->Data());
    for (size_t i = 0; i < input_num; i++) {
      input_ptr[i] = (int8_t)(input_tmp[i] - 128);
    }
    in_quant_arg_.zp_ -= 128;
    out_quant_arg_.zp_ -= 128;
  }

  size_t data_size = inputs_.at(kInputIndex)->Size();
  Reshape(input_ptr, output_ptr, data_size, input_num, in_quant_arg_, out_quant_arg_);

  auto output_type = outputs_[kOutputIndex]->data_type();
  if (output_type == kNumberTypeUInt8) {
    for (size_t i = 0; i < output_num; i++) {
      output_ptr[i] = (uint8_t)(output_ptr[i] + 128);
    }
  }
  return RET_OK;
}
}  // namespace mindspore::kernel

