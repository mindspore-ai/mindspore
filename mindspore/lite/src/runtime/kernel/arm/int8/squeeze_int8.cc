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

#include "src/runtime/kernel/arm/nnacl/int8/squeeze_int8.h"
#include "src/runtime/kernel/arm/int8/squeeze_int8.h"
#include "src/runtime/kernel/arm/nnacl/squeeze_parameter.h"

#include "schema/model_generated.h"
#include "src/runtime/runtime_api.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {

int SqueezeInt8CPUKernel::Init() {
  auto init_ret = SqueezeBaseCPUKernel::Init();
  if (init_ret != RET_OK) {
    return init_ret;
  }
  quant_Squeeze_parm_ = new (std::nothrow) SqueezeQuantArg;
  auto input_num = in_tensors_.size();
  quant_Squeeze_parm_->input_num_ = input_num;
  quant_Squeeze_parm_->input_sizes_ = reinterpret_cast<int *>(malloc(sizeof(int) * input_num));
  if (quant_Squeeze_parm_->input_sizes_ == nullptr) {
    MS_LOG(ERROR) << "Null pointer reference: quant_Squeeze_parm_->input_sizes_.";
    return RET_ERROR;
  }

  for (size_t i = 0; i < input_num; i++) {
    quant_Squeeze_parm_->input_sizes_[i] = 1;
  }
  quant_Squeeze_parm_->input_shapes_ = reinterpret_cast<int **>(malloc(sizeof(int *) * input_num));
  if (quant_Squeeze_parm_->input_shapes_ == nullptr) {
    MS_LOG(ERROR) << "Null pointer reference: quant_Squeeze_parm_->input_shapes_.";
    return RET_ERROR;
  }

  quant_Squeeze_parm_->axis_ = 0;
  quant_Squeeze_parm_->in_quant_args_ = reinterpret_cast<QuantArg *>(malloc(sizeof(QuantArg) * input_num));
  if (quant_Squeeze_parm_->in_quant_args_ == nullptr) {
    MS_LOG(ERROR) << "Null pointer reference: quant_Squeeze_parm_->in_quant_args_.";
    return RET_ERROR;
  }

  for (size_t i = 0; i < input_num; i++) {
    auto *input_tensor = in_tensors_.at(i);
    auto quant_args = input_tensor->GetQuantParams();
    MS_ASSERT(quant_args.size() == 1);
    quant_Squeeze_parm_->in_quant_args_[i].scale_ = quant_args.front().scale;
    quant_Squeeze_parm_->in_quant_args_[i].zp_ = quant_args.front().zeroPoint;
  }

  MS_ASSERT(outputs_.size() == 1);
  auto output_tensor = out_tensors_.at(0);
  MS_ASSERT(output_tensor != nullptr);
  auto quant_args = output_tensor->GetQuantParams();
  MS_ASSERT(quant_args.size() == 1);
  quant_Squeeze_parm_->out_quant_args_.scale_ = quant_args.front().scale;
  quant_Squeeze_parm_->out_quant_args_.zp_ = quant_args.front().zeroPoint;
  if (!InferShapeDone()) {
    return RET_OK;
  }

  return ReSize();
}

int SqueezeInt8CPUKernel::ReSize() {
  auto input_num = in_tensors_.size();
  for (size_t i = 0; i < input_num; i++) {
    auto *input_tensor = in_tensors_.at(i);
    MS_ASSERT(input_tensor != nullptr);
    auto input_size = input_tensor->shape().size();
    MS_ASSERT(input_size != NULL);
    quant_Squeeze_parm_->input_shapes_[i] = reinterpret_cast<int *>(malloc(sizeof(int) * input_size));
    if (quant_Squeeze_parm_->input_shapes_[i] == nullptr) {
      MS_LOG(ERROR) << "Null pointer reference: quant_Squeeze_parm_->input_shapes_[" << i << "].";
      return RET_ERROR;
    }

    ::memcpy(quant_Squeeze_parm_->input_shapes_[i], input_tensor->shape().data(), sizeof(int) * input_size);
    for (size_t j = 0; j < input_size; j++) {
      auto *input_tensor_tmp = in_tensors_.at(i);
      auto input_shape = input_tensor_tmp->shape()[j];
      quant_Squeeze_parm_->input_sizes_[i] *= input_shape;
    }
  }

  MS_ASSERT(out_tensors_.size() == 1);
  auto output_tensor = out_tensors_.at(0);
  MS_ASSERT(output_tensor != nullptr);
  auto output_shape = output_tensor->shape();
  MS_ASSERT(output_shape != NULL);
  auto output_dim = output_shape.size();
  quant_Squeeze_parm_->output_dim_ = output_dim;
  int output_size = 1;
  for (size_t i = 0; i < output_dim; i++) {
    output_size *= output_shape[i];
  }
  quant_Squeeze_parm_->output_size_ = output_size;

  quant_Squeeze_parm_->output_shape_ = new int[output_size];
  ::memcpy(quant_Squeeze_parm_->output_shape_, output_shape.data(), sizeof(int) * output_size);
  return RET_OK;
}

int SqueezeInt8CPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail!ret: " << ret;
    return ret;
  }
  auto input_dim = quant_Squeeze_parm_->input_num_;
  int8_t **inputs_array = reinterpret_cast<int8_t **>(malloc(sizeof(int8_t *) * input_dim));
  for (size_t i = 0; i < input_dim; i++) {
    auto input_size = quant_Squeeze_parm_->input_sizes_[i];
    inputs_array[i] = reinterpret_cast<int8_t *>(malloc(sizeof(int8_t) * input_size));
    auto input_type = in_tensors_[i]->data_type();
    if (input_type == kNumberTypeUInt8) {
      uint8_t *input_tmp = reinterpret_cast<uint8_t *>(in_tensors_[i]->Data());
      for (size_t j = 0; j < input_size; j++) {
        inputs_array[i][j] = (int8_t)(input_tmp[j] - 128);
      }
      for (size_t j = 0; j < input_dim; j++) {
        quant_Squeeze_parm_->in_quant_args_[j].zp_ -= 128;
      }
      quant_Squeeze_parm_->out_quant_args_.zp_ -= 128;
    } else {
      ::memcpy(inputs_array[i], in_tensors_.at(i)->Data(), sizeof(int8_t) * input_size);
    }
  }
  int8_t *output_addr = reinterpret_cast<int8_t *>(out_tensors_.at(0)->Data());
  auto output_type = out_tensors_[0]->data_type();
  if (output_type == kNumberTypeUInt8) {
    auto output_size = quant_Squeeze_parm_->output_size_;
    for (size_t i = 0; i < output_size; i++) {
      output_addr[i] = (uint8_t)(output_addr[i] + 128);
    }
  }

  for (int i = 0; i < input_dim; i++) {
    free(*(inputs_array + i));
  }

  ret = LiteBackendParallelLaunch(SqueezeInt8Run, this, thread_count_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "RunSqueezeParam failed. errorcode: ";
  }
  return ret;
}

int SqueezeInt8Run(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto Squeeze = reinterpret_cast<SqueezeInt8CPUKernel *>(cdata);
  Squeeze->DoExecute(task_id);
  return RET_OK;
}

int SqueezeInt8CPUKernel::DoExecute(int task_id) {
  auto input_tensor = in_tensors_.at(kInputIndex);
  auto out_tensor = out_tensors_.at(kOutputIndex);
  int8_t *input_data = reinterpret_cast<int8_t *>(input_tensor->Data());
  int8_t *output_data = reinterpret_cast<int8_t *>(out_tensor->Data());

  size_t data_size = in_tensors_.front()->Size();
  Squeeze(&input_data, output_data, task_id, quant_Squeeze_parm_, para_, data_size);
  return RET_OK;
}

}  // namespace mindspore::kernel
