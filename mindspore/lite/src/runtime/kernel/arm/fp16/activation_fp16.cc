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

#include "src/runtime/kernel/arm/fp16/activation_fp16.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"
#include "include/errorcode.h"
#include "src/runtime/kernel/arm/fp16/common_fp16.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::ActivationType_GELU;
using mindspore::schema::ActivationType_HSWISH;
using mindspore::schema::ActivationType_LEAKY_RELU;
using mindspore::schema::ActivationType_RELU;
using mindspore::schema::ActivationType_RELU6;
using mindspore::schema::ActivationType_SWISH;
using mindspore::schema::PrimitiveType_Activation;

namespace mindspore::kernel {
int ActivationFp16CPUKernel::Init() {
  if (type_ != schema::ActivationType_RELU && type_ != schema::ActivationType_RELU6 &&
      type_ != schema::ActivationType_LEAKY_RELU && type_ != schema::ActivationType_SIGMOID &&
      type_ != schema::ActivationType_TANH && type_ != schema::ActivationType_HSWISH &&
      type_ != schema::ActivationType_SWISH && type_ != schema::ActivationType_HARD_TANH) {
    MS_LOG(ERROR) << "Activation fp16 not support type: " << type_;
    return RET_ERROR;
  }
  return RET_OK;
}

int ActivationFp16CPUKernel::ReSize() { return RET_OK; }

int ActivationFp16CPUKernel::DoActivation(int task_id) {
  auto length = in_tensors_.at(0)->ElementsNum();

  int stride = UP_DIV(length, thread_count_);
  int count = MSMIN(stride, length - stride * task_id);
  if (count <= 0) {
    return RET_OK;
  }

  int error_code;
  if (type_ == schema::ActivationType_RELU) {
    error_code = ReluFp16(fp16_input_ + stride * task_id, fp16_output_ + stride * task_id, count);
  } else if (type_ == schema::ActivationType_RELU6) {
    error_code = Relu6Fp16(fp16_input_ + stride * task_id, fp16_output_ + stride * task_id, count);
  } else if (type_ == schema::ActivationType_LEAKY_RELU) {
    error_code = LReluFp16(fp16_input_ + stride * task_id, fp16_output_ + stride * task_id, count, alpha_);
  } else if (type_ == schema::ActivationType_SIGMOID) {
    error_code = SigmoidFp16(fp16_input_ + stride * task_id, fp16_output_ + stride * task_id, count);
  } else if (type_ == schema::ActivationType_TANH) {
    error_code = TanhFp16(fp16_input_ + stride * task_id, fp16_output_ + stride * task_id, count);
  } else if (type_ == schema::ActivationType_HSWISH) {
    error_code = HSwishFp16(fp16_input_ + stride * task_id, fp16_output_ + stride * task_id, count);
  } else if (type_ == schema::ActivationType_SWISH) {
    error_code = SwishFp16(fp16_input_ + stride * task_id, fp16_output_ + stride * task_id, count);
  } else if (type_ == schema::ActivationType_HARD_TANH) {
    error_code =
      HardTanhFp16(fp16_input_ + stride * task_id, count, fp16_output_ + stride * task_id, min_val_, max_val_);
  } else if (type_ == schema::ActivationType_GELU) {
    error_code = GeluFp16(fp16_input_ + stride * task_id, count, fp16_output_ + stride * task_id, true);
  } else {
    MS_LOG(ERROR) << "Activation fp16 not support type: " << type_;
    return RET_ERROR;
  }
  return error_code;
}

int ActivationFp16Run(void *cdata, int task_id) {
  auto activation_kernel = reinterpret_cast<ActivationFp16CPUKernel *>(cdata);
  auto error_code = activation_kernel->DoActivation(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "ActivationRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ActivationFp16CPUKernel::Run() {
  auto input_tensor = in_tensors_.at(0);
  auto output_tensor = out_tensors_.at(0);

  fp16_input_ = reinterpret_cast<float16_t *>(input_tensor->data_c());
  fp16_output_ = reinterpret_cast<float16_t *>(output_tensor->data_c());

  int error_code = ParallelLaunch(this->context_->thread_pool_, ActivationFp16Run, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Activation function error error_code[" << error_code << "]";
    return RET_ERROR;
  }

  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Activation, LiteKernelCreator<ActivationFp16CPUKernel>)
}  // namespace mindspore::kernel
