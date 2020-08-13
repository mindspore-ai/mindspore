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
#include "src/runtime/kernel/arm/int8/argminmax_int8.h"
#include <vector>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/kernel/arm/nnacl/int8/arg_min_max_int8.h"
#include "include/errorcode.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int ArgMinMaxInt8CPUKernel::Init() {
  auto ret = ArgMinMaxBaseCPUKernel::Init();
  if (ret != RET_OK) {
    return ret;
  }
  auto param = reinterpret_cast<ArgMinMaxParameter *>(op_parameter_);
  param->data_type_ = kNumberTypeInt8;
  auto *input_tensor = in_tensors_.at(kInputIndex);
  auto in_quant_args = input_tensor->GetQuantParams();
  in_quant_arg_.scale_ = in_quant_args.front().scale;
  in_quant_arg_.zp_ = in_quant_args.front().zeroPoint;

  auto *out_tensor = out_tensors_.at(kOutputIndex);
  auto out_quant_args = out_tensor->GetQuantParams();
  out_quant_arg_.scale_ = out_quant_args.front().scale;
  out_quant_arg_.zp_ = out_quant_args.front().zeroPoint;
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ArgMinMaxInt8CPUKernel::ReSize() { return ArgMinMaxBaseCPUKernel::ReSize(); }

int ArgMinMaxInt8CPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << ret;
    return ret;
  }
  auto input = in_tensors_.at(0);

  const int8_t *input_data = reinterpret_cast<const int8_t *>(in_tensors_.at(0)->Data());
  int8_t *output_data = reinterpret_cast<int8_t *>(out_tensors_.at(0)->Data());

  auto in_shape = input->shape().data();
  auto param = reinterpret_cast<ArgMinMaxParameter *>(op_parameter_);
  if (param->topk_ == 1) {
    Int8ArgMinMaxQuant(input_data, output_data, in_shape, param, &in_quant_arg_, &out_quant_arg_);
    return RET_OK;
  }

  switch (param->axis_) {
    case 0:
      Int8ArgMinMaxDim0(input_data, output_data, in_shape, param, &in_quant_arg_, &out_quant_arg_);
      break;
    case 1:
      Int8ArgMinMaxDim1(input_data, output_data, in_shape, param, &in_quant_arg_, &out_quant_arg_);
      break;
    case 2:
      Int8ArgMinMaxDim2(input_data, output_data, in_shape, param, &in_quant_arg_, &out_quant_arg_);
      break;
    case 3:
      Int8ArgMinMaxDim3(input_data, output_data, in_shape, param, &in_quant_arg_, &out_quant_arg_);
      break;
  }
  return RET_OK;
}
}  // namespace mindspore::kernel
