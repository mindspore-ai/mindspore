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
#include "schema/model_generated.h"
#include "src/kernel_registry.h"

using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_FORMAT_ERR;
using mindspore::lite::RET_PARAM_INVALID;
using mindspore::schema::PrimitiveType_ArgMaxFusion;
using mindspore::schema::PrimitiveType_ArgMinFusion;

namespace mindspore::kernel {
int ArgMinMaxInt8CPUKernel::Init() {
  auto param = reinterpret_cast<ArgMinMaxParameter *>(op_parameter_);
  param->data_type_ = kNumberTypeInt8;
  auto *input_tensor = in_tensors_.at(kInputIndex);
  auto in_quant_args = input_tensor->quant_params();
  in_quant_arg_.scale_ = in_quant_args.front().scale;
  in_quant_arg_.zp_ = in_quant_args.front().zeroPoint;

  auto *out_tensor = out_tensors_.at(kOutputIndex);
  auto out_quant_args = out_tensor->quant_params();
  out_quant_arg_.scale_ = out_quant_args.front().scale;
  out_quant_arg_.zp_ = out_quant_args.front().zeroPoint;
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ArgMinMaxInt8CPUKernel::ReSize() {
  auto in_shape = in_tensors_.at(0)->shape();
  auto dims_size = in_shape.size();
  auto param = reinterpret_cast<ArgMinMaxParameter *>(op_parameter_);
  int axis = param->axis_ < 0 ? param->axis_ + dims_size : param->axis_;
  param->axis_ = axis;
  param->dims_size_ = dims_size;
  if (param->topk_ <= 0) {
    MS_LOG(ERROR) << "Invalid topk " << param->topk_;
    return RET_ERROR;
  }
  param->topk_ = MSMIN(param->topk_, in_shape.at(axis));
  ComputeStrides(in_shape.data(), param->in_strides_, in_shape.size());
  auto out_shape = out_tensors_.at(0)->shape();
  ComputeStrides(out_shape.data(), param->out_strides_, out_shape.size());
  return RET_OK;
}

int ArgMinMaxInt8CPUKernel::Run() {
  auto input = in_tensors_.at(0);

  const int8_t *input_data = reinterpret_cast<const int8_t *>(in_tensors_.at(0)->MutableData());
  int8_t *output_data = reinterpret_cast<int8_t *>(out_tensors_.at(0)->MutableData());

  auto in_shape = input->shape();
  auto param = reinterpret_cast<ArgMinMaxParameter *>(op_parameter_);
  if (param->topk_ == 1) {
    Int8ArgMinMaxQuant(input_data, output_data, in_shape.data(), param, &in_quant_arg_, &out_quant_arg_);
    return RET_OK;
  }

  switch (param->axis_) {
    case 0:
      Int8ArgMinMaxDim0(input_data, output_data, in_shape.data(), param, &in_quant_arg_, &out_quant_arg_);
      break;
    case 1:
      Int8ArgMinMaxDim1(input_data, output_data, in_shape.data(), param, &in_quant_arg_, &out_quant_arg_);
      break;
    case 2:
      Int8ArgMinMaxDim2(input_data, output_data, in_shape.data(), param, &in_quant_arg_, &out_quant_arg_);
      break;
    case 3:
      Int8ArgMinMaxDim3(input_data, output_data, in_shape.data(), param, &in_quant_arg_, &out_quant_arg_);
      break;
    default:
      MS_LOG(ERROR) << "axis is invalid";
      return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_ArgMaxFusion, LiteKernelCreator<ArgMinMaxInt8CPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_ArgMinFusion, LiteKernelCreator<ArgMinMaxInt8CPUKernel>)
}  // namespace mindspore::kernel
