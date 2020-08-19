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
#include "nnacl/fp16/cast_fp16.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::ActivationType_HSWISH;
using mindspore::schema::ActivationType_LEAKY_RELU;
using mindspore::schema::ActivationType_RELU;
using mindspore::schema::ActivationType_RELU6;
using mindspore::schema::PrimitiveType_Activation;

namespace mindspore::kernel {
int ActivationFp16CPUKernel::Init() { return RET_OK; }

int ActivationFp16CPUKernel::ReSize() { return RET_OK; }

int ActivationFp16CPUKernel::MallocTmpBuffer() {
  fp16_input_  = ConvertInputFp32toFp16(in_tensors_.at(0), context_);
  if (fp16_input_ == nullptr) {
    MS_LOG(ERROR) << "malloc data failed";
    return RET_ERROR;
  }
  fp16_output_ = MallocOutputFp16(out_tensors_.at(0), context_);
  if (fp16_output_ == nullptr) {
    MS_LOG(ERROR) << "malloc data failed";
    return RET_ERROR;
  }
  return RET_OK;
}

void ActivationFp16CPUKernel::FreeTmpBuffer() {
  if (in_tensors_.at(0)->data_type() == kNumberTypeFloat32) {
    if (fp16_input_ != nullptr) {
      context_->allocator->Free(fp16_input_);
      fp16_input_ = nullptr;
    }
  }
  if (out_tensors_.at(0)->data_type() == kNumberTypeFloat32) {
    if (fp16_output_ != nullptr) {
      context_->allocator->Free(fp16_output_);
      fp16_output_ = nullptr;
    }
  }
}

int ActivationFp16CPUKernel::DoActivation(int task_id) {
  auto length = in_tensors_.at(0)->ElementsNum();

  int stride = UP_DIV(length, thread_count_);
  int count = MSMIN(stride, length - stride * task_id);

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
  } else {
    MS_LOG(ERROR) << "Activation fp16 not support type: " << type_;
    return RET_ERROR;
  }
  return error_code;
}

int ActivationRun(void *cdata, int task_id) {
  auto activation_kernel = reinterpret_cast<ActivationFp16CPUKernel *>(cdata);
  auto error_code = activation_kernel->DoActivation(task_id);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "ActivationRun error task_id[" << task_id << "] error_code[" << error_code << "]";
    return RET_ERROR;
  }
  return RET_OK;
}

int ActivationFp16CPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare failed.";
    return ret;
  }

  ret = MallocTmpBuffer();
  if (ret != RET_OK) {
    FreeTmpBuffer();
    return ret;
  }

  int error_code = ParallelLaunch(THREAD_POOL_DEFAULT, ActivationRun, this, thread_count_);
  if (error_code != RET_OK) {
    MS_LOG(ERROR) << "Activation function error error_code[" << error_code << "]";
    FreeTmpBuffer();
    return RET_ERROR;
  }

  auto out_tensor = out_tensors_.at(0);
  if (out_tensor->data_type() == kNumberTypeFloat32) {
    Float16ToFloat32(fp16_output_, reinterpret_cast<float *>(out_tensor->Data()), out_tensor->ElementsNum());
  }
  FreeTmpBuffer();
  return RET_OK;
}

kernel::LiteKernel *CpuActivationFp16KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                   const std::vector<lite::tensor::Tensor *> &outputs,
                                                   OpParameter *opParameter, const lite::Context *ctx,
                                                   const kernel::KernelKey &desc,
                                                   const mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_Activation);
  auto *kernel = new (std::nothrow) ActivationFp16CPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "kernel is nullptr.";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    delete kernel;
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimitiveType_Activation, CpuActivationFp16KernelCreator)
}  // namespace mindspore::kernel
