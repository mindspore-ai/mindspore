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
#include "src/litert/kernel/cpu/int8/depth_to_space_int8.h"
#include <cfloat>
#include <cmath>
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_FORMAT_ERR;
using mindspore::lite::RET_OK;
using mindspore::lite::RET_PARAM_INVALID;
using mindspore::schema::PrimitiveType_DepthToSpace;

namespace mindspore::kernel {
DepthToSpaceInt8CPUKernel::~DepthToSpaceInt8CPUKernel() {
  if (in_quant_arg_ != nullptr) {
    free(in_quant_arg_);
    in_quant_arg_ = nullptr;
  }
  if (out_quant_arg_ != nullptr) {
    free(out_quant_arg_);
    out_quant_arg_ = nullptr;
  }
}

int DepthToSpaceInt8CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(in_tensors_[0]);
  CHECK_NULL_RETURN(out_tensors_[0]);
  if (in_tensors_[0]->data_type() != mindspore::kNumberTypeInt8 ||
      out_tensors_[0]->data_type() != mindspore::kNumberTypeInt8) {
    MS_LOG(ERROR) << "Datatype error, input0 data_type is " << in_tensors_[0]->data_type() << ", output data_type is "
                  << out_tensors_[0]->data_type();
    return RET_ERROR;
  }
  param_->data_type_size_ = sizeof(int8_t);

  in_quant_arg_ = reinterpret_cast<QuantArg *>(malloc(sizeof(QuantArg)));
  if (in_quant_arg_ == nullptr) {
    MS_LOG(ERROR) << "Malloc QuantArg for DepthToSpace int8 op failed!";
    return RET_ERROR;
  }
  auto *input_tensor = in_tensors_.at(kInputIndex);
  auto in_quant_args = input_tensor->quant_params();
  CHECK_LESS_RETURN(in_quant_args.size(), 1);
  in_quant_arg_->scale_ = in_quant_args.front().scale;
  in_quant_arg_->zp_ = in_quant_args.front().zeroPoint;

  out_quant_arg_ = reinterpret_cast<QuantArg *>(malloc(sizeof(QuantArg)));
  if (out_quant_arg_ == nullptr) {
    MS_LOG(ERROR) << "Malloc QuantArg for DepthToSpace int8 op failed!";
    return RET_ERROR;
  }
  auto *out_tensor = out_tensors_.at(kOutputIndex);
  auto out_quant_args = out_tensor->quant_params();
  CHECK_LESS_RETURN(out_quant_args.size(), 1);
  out_quant_arg_->scale_ = out_quant_args.front().scale;
  out_quant_arg_->zp_ = out_quant_args.front().zeroPoint;
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int DepthToSpaceInt8CPUKernel::Run() {
  auto input = in_tensors_[0];
  auto output = out_tensors_[0];
  const int8_t *input_data = reinterpret_cast<const int8_t *>(input->data());
  CHECK_NULL_RETURN(input_data);
  int8_t *output_data = reinterpret_cast<int8_t *>(output->data());
  CHECK_NULL_RETURN(output_data);
  auto in_shape = input->shape();
  if (std::abs(in_quant_arg_->scale_ - out_quant_arg_->scale_) < FLT_EPSILON &&
      in_quant_arg_->zp_ == out_quant_arg_->zp_) {
    DepthToSpaceForNHWC(input_data, output_data, in_shape.data(), param_);
  } else {
    DepthToSpaceForNHWCInt8(input_data, output_data, in_shape.data(), param_, in_quant_arg_, out_quant_arg_);
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_DepthToSpace, LiteKernelCreator<DepthToSpaceInt8CPUKernel>)
}  // namespace mindspore::kernel
