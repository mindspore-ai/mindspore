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

#include "src/runtime/kernel/arm/int8/concat_int8.h"
#include "src/runtime/kernel/arm/nnacl/int8/concat_int8.h"
#include "schema/model_generated.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {

int ConcatInt8CPUKernel::Init() {
  ConcatBaseCPUKernel::Init();
  quant_concat_parm_ = concat_param_->concat_quant_arg_;
  quant_concat_parm_ = new (std::nothrow) ConcatQuantArg;
  auto input_num = inputs_.size();
  quant_concat_parm_->input_num_ = input_num;
  quant_concat_parm_->input_sizes_ = reinterpret_cast<int *>(malloc(sizeof(int) * input_num));
  if (quant_concat_parm_->input_sizes_ == nullptr) {
    MS_LOG(ERROR) << "Null pointer reference: quant_concat_parm_->input_sizes_.";
    return RET_ERROR;
  }

  for (size_t i = 0; i < input_num; i++) {
    quant_concat_parm_->input_sizes_[i] = 1;
  }
  quant_concat_parm_->input_shapes_ = reinterpret_cast<int **>(malloc(sizeof(int *) * input_num));
  if (quant_concat_parm_->input_shapes_ == nullptr) {
    MS_LOG(ERROR) << "Null pointer reference: quant_concat_parm_->input_shapes_.";
    return RET_ERROR;
  }

  for (size_t i = 0; i < input_num; i++) {
    auto *input_tensor = inputs_.at(i);
    MS_ASSERT(input_tensor != nullptr);
    auto input_size = input_tensor->shape().size();
    MS_ASSERT(input_size != NULL);
    quant_concat_parm_->input_shapes_[i] = reinterpret_cast<int *>(malloc(sizeof(int) * input_size));
    if (quant_concat_parm_->input_shapes_[i] == nullptr) {
      MS_LOG(ERROR) << "Null pointer reference: quant_concat_parm_->input_shapes_[" << i << "].";
      return RET_ERROR;
    }

    ::memcpy(quant_concat_parm_->input_shapes_[i], input_tensor->shape().data(), sizeof(int) * input_size);
    for (size_t j = 0; j < input_size; j++) {
      auto *input_tensor_tmp = inputs_.at(i);
      auto input_shape = input_tensor_tmp->shape()[j];
      quant_concat_parm_->input_sizes_[i] *= input_shape;
    }
  }

  quant_concat_parm_->in_quant_args_ = reinterpret_cast<QuantArg *>(malloc(sizeof(QuantArg) * input_num));
  if (quant_concat_parm_->in_quant_args_ == nullptr) {
    MS_LOG(ERROR) << "Null pointer reference: quant_concat_parm_->in_quant_args_.";
    return RET_ERROR;
  }

  for (size_t i = 0; i < input_num; i++) {
    auto *input_tensor = inputs_.at(i);
    auto quant_args = input_tensor->GetQuantParams();
    MS_ASSERT(quant_args.size() == 1);
    quant_concat_parm_->in_quant_args_[i].scale_ = quant_args.front().scale;
    quant_concat_parm_->in_quant_args_[i].zp_ = quant_args.front().zeroPoint;
  }

  MS_ASSERT(outputs_.size() == 1);
  auto output_tensor = outputs_.at(0);
  MS_ASSERT(output_tensor != nullptr);
  auto output_shape = output_tensor->shape();
  MS_ASSERT(output_shape != NULL);
  auto output_dim = output_shape.size();
  quant_concat_parm_->output_dim_ = output_dim;
  int output_size = 1;
  for (size_t i = 0; i < output_dim; i++) {
    output_size *= output_shape[i];
  }
  quant_concat_parm_->output_size_ = output_size;

  quant_concat_parm_->output_shape_ = new int[output_size];
  ::memcpy(quant_concat_parm_->output_shape_, output_shape.data(), sizeof(int) * output_size);

  auto quant_args = output_tensor->GetQuantParams();
  MS_ASSERT(quant_args.size() == 1);
  quant_concat_parm_->out_quant_args_.scale_ = quant_args.front().scale;
  quant_concat_parm_->out_quant_args_.zp_ = quant_args.front().zeroPoint;

  return RET_OK;
}

int ConcatInt8CPUKernel::ReSize() { return 0; }

int ConcatInt8CPUKernel::Run() {
  auto input_dim = quant_concat_parm_->input_num_;
  int8_t **inputs_array = reinterpret_cast<int8_t **>(malloc(sizeof(int8_t *) * input_dim));
  for (size_t i = 0; i < input_dim; i++) {
    auto input_size = quant_concat_parm_->input_sizes_[i];
    inputs_array[i] = reinterpret_cast<int8_t *>(malloc(sizeof(int8_t) * input_size));
    auto input_type = inputs_[i]->data_type();
    if (input_type == kNumberTypeUInt8) {
      uint8_t *input_tmp = reinterpret_cast<uint8_t *>(inputs_[i]->Data());
      for (size_t j = 0; j < input_size; j++) {
        inputs_array[i][j] = (int8_t)(input_tmp[j] - 128);
      }
      for (size_t j = 0; j < input_dim; j++) {
        quant_concat_parm_->in_quant_args_[j].zp_ -= 128;
      }
      quant_concat_parm_->out_quant_args_.zp_ -= 128;
    } else {
      ::memcpy(inputs_array[i], inputs_.at(i)->Data(), sizeof(int8_t) * input_size);
    }
  }
  int8_t *output_addr = reinterpret_cast<int8_t *>(outputs_.at(0)->Data());
  Concat(inputs_array, output_addr, quant_concat_parm_, axis_);
  auto output_type = outputs_[0]->data_type();
  if (output_type == kNumberTypeUInt8) {
    auto output_size = quant_concat_parm_->output_size_;
    for (size_t i = 0; i < output_size; i++) {
      output_addr[i] = (uint8_t)(output_addr[i] + 128);
    }
  }

  for (int i = 0; i < input_dim; i++) {
    free(*(inputs_array + i));
  }
  return RET_OK;
}
}  // namespace mindspore::kernel

