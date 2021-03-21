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

#include "src/runtime/kernel/arm/fp32/activation_fp32.h"
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "src/runtime/runtime_api.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::ActivationType_HSIGMOID;
using mindspore::schema::ActivationType_HSWISH;
using mindspore::schema::ActivationType_LEAKY_RELU;
using mindspore::schema::ActivationType_RELU;
using mindspore::schema::ActivationType_RELU6;
using mindspore::schema::ActivationType_SWISH;
using mindspore::schema::PrimitiveType_Activation;

namespace mindspore::kernel {
int ActivationCPUKernel::Init() {
  if (type_ != schema::ActivationType_RELU && type_ != schema::ActivationType_RELU6 &&
      type_ != schema::ActivationType_LEAKY_RELU && type_ != schema::ActivationType_SIGMOID &&
      type_ != schema::ActivationType_TANH && type_ != schema::ActivationType_HSWISH &&
      type_ != schema::ActivationType_SWISH && type_ != schema::ActivationType_HSIGMOID &&
      type_ != schema::ActivationType_HARD_TANH && type_ != schema::ActivationType_GELU) {
    MS_LOG(ERROR) << "Activation fp32 not support type: " << type_;
    return RET_ERROR;
  }
  return RET_OK;
}

int ActivationCPUKernel::ReSize() { return RET_OK; }

int ActivationCPUKernel::DoActivation(int task_id) {
  auto input_addr = reinterpret_cast<float *>(in_tensors_.at(0)->MutableData());
  auto output_addr = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());
  auto length = in_tensors_.at(0)->ElementsNum();

  int stride = UP_DIV(length, thread_count_);
  int count = MSMIN(stride, length - stride * task_id);
  if (count <= 0) {
    return RET_OK;
  }

  auto ret = RET_OK;

  if (type_ == schema::ActivationType_RELU) {
    ret = Fp32Relu(input_addr + stride * task_id, count, output_addr + stride * task_id);
  } else if (type_ == schema::ActivationType_RELU6) {
    ret = Fp32Relu6(input_addr + stride * task_id, count, output_addr + stride * task_id);
  } else if (type_ == schema::ActivationType_LEAKY_RELU) {
    ret = LRelu(input_addr + stride * task_id, count, output_addr + stride * task_id, alpha_);
  } else if (type_ == schema::ActivationType_SIGMOID) {
    ret = Sigmoid(input_addr + stride * task_id, count, output_addr + stride * task_id);
  } else if (type_ == schema::ActivationType_TANH) {
    ret = Tanh(input_addr + stride * task_id, count, output_addr + stride * task_id);
  } else if (type_ == schema::ActivationType_SWISH) {
    ret = Swish(input_addr + stride * task_id, count, output_addr + stride * task_id);
  } else if (type_ == schema::ActivationType_HSWISH) {
    ret = HSwish(input_addr + stride * task_id, count, output_addr + stride * task_id);
  } else if (type_ == schema::ActivationType_HSIGMOID) {
    ret = HSigmoid(input_addr + stride * task_id, count, output_addr + stride * task_id);
  } else if (type_ == schema::ActivationType_HARD_TANH) {
    ret = HardTanh(input_addr + stride * task_id, count, output_addr + stride * task_id, min_val_, max_val_);
  } else if (type_ == schema::ActivationType_GELU) {
    ret = Gelu(input_addr + stride * task_id, count, output_addr + stride * task_id, true);
  } else {
    MS_LOG(ERROR) << "Activation type error";
    return RET_ERROR;
  }
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Activation error, ret: " << ret;
  }
  return ret;
}

int ActivationRun(void *cdata, int task_id) {
  auto activation_kernel = reinterpret_cast<ActivationCPUKernel *>(cdata);
  auto error_code = activation_kernel->DoActivation(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "ActivationRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ActivationCPUKernel::Run() {
  int error_code = ParallelLaunch(this->context_->thread_pool_, ActivationRun, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Activation function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Activation, LiteKernelCreator<ActivationCPUKernel>)
}  // namespace mindspore::kernel
