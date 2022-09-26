/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "src/litert/kernel/cpu/fp32/activation_fp32.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::ActivationType_HSIGMOID;
using mindspore::schema::ActivationType_HSWISH;
using mindspore::schema::ActivationType_LEAKY_RELU;
using mindspore::schema::ActivationType_RELU;
using mindspore::schema::ActivationType_RELU6;
using mindspore::schema::ActivationType_SOFTPLUS;
using mindspore::schema::ActivationType_SWISH;
using mindspore::schema::PrimitiveType_Activation;

namespace mindspore::kernel {
int ActivationCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_NOT_EQUAL_RETURN(out_tensors_.size(), 1);

  if (in_tensors().front()->data_type() == kNumberTypeInt32) {
    if (type_ != schema::ActivationType_RELU) {
      MS_LOG(ERROR) << "Activation int32 not support type: " << type_;
      return RET_ERROR;
    }
  }

  if (in_tensors().front()->data_type() == kNumberTypeFloat32) {
    if (type_ != schema::ActivationType_RELU && type_ != schema::ActivationType_RELU6 &&
        type_ != schema::ActivationType_LEAKY_RELU && type_ != schema::ActivationType_SIGMOID &&
        type_ != schema::ActivationType_TANH && type_ != schema::ActivationType_HSWISH &&
        type_ != schema::ActivationType_SWISH && type_ != schema::ActivationType_HSIGMOID &&
        type_ != schema::ActivationType_HARD_TANH && type_ != schema::ActivationType_GELU &&
        type_ != schema::ActivationType_SOFTPLUS && type_ != schema::ActivationType_ELU) {
      MS_LOG(ERROR) << "Activation fp32 not support type: " << type_;
      return RET_ERROR;
    }
  }

  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int ActivationCPUKernel::ReSize() {
  if (UpdateThreadNumPass(TC_TYPE(PrimitiveType_Activation, type_), 1, 1, out_tensors_.at(0)->ElementsNum()) !=
      RET_OK) {
    return RET_ERROR;
  }
  return RET_OK;
}

int ActivationCPUKernel::DoActivation(int task_id) {
  if (in_tensors_.front()->data_type() == kNumberTypeFloat32) {
    return DoActivationFp32(task_id);
  } else if (in_tensors_.front()->data_type() == kNumberTypeInt32) {
    return DoActivationInt32(task_id);
  }
  return RET_ERROR;
}

int ActivationCPUKernel::DoActivationInt32(int task_id) {
  auto input_addr = reinterpret_cast<int32_t *>(in_tensors_.at(0)->data());
  auto output_addr = reinterpret_cast<int32_t *>(out_tensors_.at(0)->data());
  CHECK_NULL_RETURN(input_addr);
  CHECK_NULL_RETURN(output_addr);
  auto length = in_tensors_.at(0)->ElementsNum();

  int stride = UP_DIV(length, thread_num_);
  int count = MSMIN(stride, length - stride * task_id);
  if (count <= 0) {
    return RET_OK;
  }
  if (INT_MUL_OVERFLOW(stride, task_id)) {
    return RET_ERROR;
  }

  auto ret = RET_OK;
  if (type_ == schema::ActivationType_RELU) {
    ret = Int32Relu(input_addr + stride * task_id, count, output_addr + stride * task_id);
  } else {
    MS_LOG(ERROR) << "Int32 Activation type error";
    return RET_ERROR;
  }
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Int32 Activation error, ret: " << ret;
  }
  return ret;
}

int ActivationCPUKernel::DoActivationFp32(int task_id) {
  auto input_addr = reinterpret_cast<float *>(in_tensors_.at(0)->MutableData());
  auto output_addr = reinterpret_cast<float *>(out_tensors_.at(0)->MutableData());
  CHECK_NULL_RETURN(input_addr);
  CHECK_NULL_RETURN(output_addr);
  auto length = in_tensors_.at(0)->ElementsNum();

  int stride = UP_DIV(length, thread_num_);
  int count = MSMIN(stride, length - stride * task_id);
  if (count <= 0) {
    return RET_OK;
  }
  if (INT_MUL_OVERFLOW(stride, task_id)) {
    return RET_ERROR;
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
    ret = Gelu(input_addr + stride * task_id, count, output_addr + stride * task_id, approximate_);
  } else if (type_ == schema::ActivationType_SOFTPLUS) {
    ret = Softplus(input_addr + stride * task_id, count, output_addr + stride * task_id);
  } else if (type_ == schema::ActivationType_ELU) {
    ret = Elu(input_addr + stride * task_id, count, output_addr + stride * task_id, alpha_);
  } else {
    MS_LOG(ERROR) << "Fp32 Activation type error";
    return RET_ERROR;
  }
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Fp32 Activation error, ret: " << ret;
  }
  return ret;
}

int ActivationRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  CHECK_NULL_RETURN(cdata);
  auto activation_kernel = reinterpret_cast<ActivationCPUKernel *>(cdata);
  MS_ASSERT(activation_kernel != nullptr);
  auto error_code = activation_kernel->DoActivation(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "ActivationRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ActivationCPUKernel::Run() {
  int error_code = ParallelLaunch(this->ms_context_, ActivationRun, this, thread_num_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Activation function error error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_Activation, LiteKernelCreator<ActivationCPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt32, PrimitiveType_Activation, LiteKernelCreator<ActivationCPUKernel>)
}  // namespace mindspore::kernel
