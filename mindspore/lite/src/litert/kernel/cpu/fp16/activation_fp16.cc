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

#include "src/litert/kernel/cpu/fp16/activation_fp16.h"
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"
#include "src/litert/kernel/cpu/fp16/common_fp16.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::ActivationType_GELU;
using mindspore::schema::ActivationType_HSIGMOID;
using mindspore::schema::ActivationType_HSWISH;
using mindspore::schema::ActivationType_LEAKY_RELU;
using mindspore::schema::ActivationType_RELU;
using mindspore::schema::ActivationType_RELU6;
using mindspore::schema::ActivationType_SOFTPLUS;
using mindspore::schema::ActivationType_SWISH;
using mindspore::schema::PrimitiveType_Activation;

namespace mindspore::kernel {
int ActivationFp16CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), 1);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
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
  if (INT_MUL_OVERFLOW(stride, task_id)) {
    MS_LOG(ERROR) << "int mul overflow.";
    return RET_ERROR;
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
  } else if (type_ == schema::ActivationType_HSIGMOID) {
    error_code = HSigmoidFp16(fp16_input_ + stride * task_id, fp16_output_ + stride * task_id, count);
  } else if (type_ == schema::ActivationType_HARD_TANH) {
    error_code =
      HardTanhFp16(fp16_input_ + stride * task_id, count, fp16_output_ + stride * task_id, min_val_, max_val_);
  } else if (type_ == schema::ActivationType_GELU) {
    error_code = GeluFp16(fp16_input_ + stride * task_id, count, fp16_output_ + stride * task_id, true);
  } else if (type_ == schema::ActivationType_SOFTPLUS) {
    error_code = SoftplusFp16(fp16_input_ + stride * task_id, count, fp16_output_ + stride * task_id);
  } else if (type_ == schema::ActivationType_ELU) {
    error_code = EluFp16(fp16_input_ + stride * task_id, count, fp16_output_ + stride * task_id, alpha_);
  } else {
    MS_LOG(ERROR) << "Activation fp16 not support type: " << type_;
    return RET_ERROR;
  }
  return error_code;
}

int ActivationFp16Run(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  CHECK_NULL_RETURN(cdata);
  auto activation_kernel = reinterpret_cast<ActivationFp16CPUKernel *>(cdata);
  MS_ASSERT(activation_kernel != nullptr);
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
  MS_ASSERT(input_tensor != nullptr);
  MS_ASSERT(output_tensor != nullptr);

  fp16_input_ = reinterpret_cast<float16_t *>(input_tensor->data());
  fp16_output_ = reinterpret_cast<float16_t *>(output_tensor->data());

  int error_code = ParallelLaunch(this->ms_context_, ActivationFp16Run, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Activation function error error_code[" << error_code << "]";
    return RET_ERROR;
  }

  return RET_OK;
}

/* creator func */
kernel::LiteKernel *CpuActivationFp16KernelCreator(const std::vector<lite::Tensor *> &inputs,
                                                   const std::vector<lite::Tensor *> &outputs, OpParameter *opParameter,
                                                   const lite::InnerContext *ctx, const kernel::KernelKey &desc) {
  auto act_param = reinterpret_cast<ActivationParameter *>(opParameter);
  auto type = act_param->type_;
  if (type != schema::ActivationType_RELU && type != schema::ActivationType_RELU6 &&
      type != schema::ActivationType_LEAKY_RELU && type != schema::ActivationType_SIGMOID &&
      type != schema::ActivationType_TANH && type != schema::ActivationType_HSWISH &&
      type != schema::ActivationType_SWISH && type != schema::ActivationType_HARD_TANH &&
      type != schema::ActivationType_GELU && type != schema::ActivationType_HSIGMOID &&
      type != schema::ActivationType_SOFTPLUS && type != schema::ActivationType_ELU) {
    MS_LOG(ERROR) << "Activation fp16 not support type: " << type;
    free(opParameter);
    return nullptr;
  }

  kernel::LiteKernel *kernel = new (std::nothrow) kernel::ActivationFp16CPUKernel(opParameter, inputs, outputs, ctx);
  if (kernel == nullptr) {
    MS_LOG(DEBUG) << "Create activation fp16 kernel failed.";
    free(opParameter);
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Activation, CpuActivationFp16KernelCreator)
}  // namespace mindspore::kernel
