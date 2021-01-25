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

#include "src/runtime/kernel/arm/int8/leaky_relu_int8.h"
#include "src/runtime/runtime_api.h"
#include "src/kernel_registry.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_MEMORY_FAILED;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_LeakyRelu;

namespace mindspore::kernel {
namespace {
int LeakyReluInt8Run(void *cdata, int task_id) {
  if (cdata == nullptr) {
    MS_LOG(ERROR) << "input cdata is nullptr!";
    return RET_ERROR;
  }
  auto relu = reinterpret_cast<LeakyReluInt8CPUKernel *>(cdata);
  relu->DoExecute(task_id);
  return RET_OK;
}
}  // namespace

int LeakyReluInt8CPUKernel::Init() {
  quant_prelu_parm_.op_parameter_ = *op_parameter_;
  quant_prelu_parm_.slope_ = reinterpret_cast<ActivationParameter *>(op_parameter_)->alpha_;

  auto *input_tensor = in_tensors_.at(kInputIndex);
  auto in_quant_args = input_tensor->quant_params();
  quant_prelu_parm_.quant_arg.in_args_.scale_ = in_quant_args.front().scale;
  quant_prelu_parm_.quant_arg.in_args_.zp_ = in_quant_args.front().zeroPoint;

  auto *out_tensor = out_tensors_.at(kOutputIndex);
  auto out_quant_args = out_tensor->quant_params();
  quant_prelu_parm_.quant_arg.out_args_.scale_ = out_quant_args.front().scale;
  quant_prelu_parm_.quant_arg.out_args_.zp_ = out_quant_args.front().zeroPoint;

  quant_prelu_parm_.quant_arg.output_activation_max_ = std::numeric_limits<int8_t>::max();
  quant_prelu_parm_.quant_arg.output_activation_min_ = std::numeric_limits<int8_t>::min();

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

LeakyReluInt8CPUKernel::~LeakyReluInt8CPUKernel() {
  if (quant_prelu_parm_.in_shape_ != nullptr) {
    free(quant_prelu_parm_.in_shape_);
    quant_prelu_parm_.in_shape_ = nullptr;
  }
  if (quant_prelu_parm_.out_shape_ != nullptr) {
    free(quant_prelu_parm_.out_shape_);
    quant_prelu_parm_.out_shape_ = nullptr;
  }
}

int LeakyReluInt8CPUKernel::ReSize() {
  auto *input_tensor = in_tensors_.at(kInputIndex);
  auto *out_tensor = out_tensors_.at(kOutputIndex);
  auto input_dim = input_tensor->shape().size();
  quant_prelu_parm_.input_dim_ = input_dim;
  quant_prelu_parm_.element_num = in_tensors_.at(0)->Size();
  auto input_shape = input_tensor->shape();
  if (quant_prelu_parm_.in_shape_ != nullptr) {
    free(quant_prelu_parm_.in_shape_);
    quant_prelu_parm_.in_shape_ = nullptr;
  }
  quant_prelu_parm_.in_shape_ = reinterpret_cast<int *>(malloc(input_shape.size() * sizeof(int)));
  if (quant_prelu_parm_.in_shape_ == nullptr) {
    MS_LOG(ERROR) << "malloc memory failed";
    return RET_MEMORY_FAILED;
  }
  memcpy(reinterpret_cast<void *>(const_cast<int *>(quant_prelu_parm_.in_shape_)), input_shape.data(),
         sizeof(int) * input_dim);

  auto output_shape = out_tensor->shape();
  size_t output_dim = output_shape.size();
  if (quant_prelu_parm_.out_shape_ != nullptr) {
    free(quant_prelu_parm_.out_shape_);
    quant_prelu_parm_.out_shape_ = nullptr;
  }
  quant_prelu_parm_.out_shape_ = reinterpret_cast<int *>(malloc(output_dim * sizeof(int)));
  if (quant_prelu_parm_.out_shape_ == nullptr) {
    MS_LOG(ERROR) << "malloc memory failed";
    return RET_MEMORY_FAILED;
  }
  memcpy(reinterpret_cast<void *>(const_cast<int *>(quant_prelu_parm_.out_shape_)), output_shape.data(),
         sizeof(int) * output_dim);
  return RET_OK;
}

int LeakyReluInt8CPUKernel::Run() {
  auto ret = ParallelLaunch(this->context_->thread_pool_, LeakyReluInt8Run, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "RunPreluParam failed. errorcode: ";
  }
  return RET_OK;
}

int LeakyReluInt8CPUKernel::DoExecute(int task_id) {
  auto input_tensor = in_tensors_.at(kInputIndex);
  auto out_tensor = out_tensors_.at(kOutputIndex);
  int8_t *input_data = reinterpret_cast<int8_t *>(input_tensor->data_c());
  int8_t *output_data = reinterpret_cast<int8_t *>(out_tensor->data_c());
  MS_ASSERT(input_data);
  MS_ASSERT(output_data);
  auto ret = DoLeakReluInt8(input_data, output_data, &quant_prelu_parm_, task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "DoLeakReluInt8 failed";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_LeakyRelu, LiteKernelCreator<LeakyReluInt8CPUKernel>)
}  // namespace mindspore::kernel
