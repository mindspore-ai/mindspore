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

#include "src/runtime/kernel/arm/fp32/batchnorm.h"
#include "src/kernel_registry.h"

using mindspore::lite::KernelRegistrar;
using mindspore::schema::PrimitiveType_BatchNorm;

namespace mindspore::kernel {
int BatchnormCPUKernel::Init() {
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int BatchnormCPUKernel::ReSize() {
  FreeMeanAndVariance();
  FillParam();
  return InitConstTensor();
}

void BatchnormCPUKernel::FreeMeanAndVariance() {
  if (mean_ != nullptr) {
    free(mean_);
    mean_ = nullptr;
  }
  if (variance_ != nullptr) {
    free(variance_);
    variance_ = nullptr;
  }
}

void BatchnormCPUKernel::FillParam() {
  auto input_shapes = in_tensors_[0]->shape();
  auto n_dim = input_shapes.size();
  auto param = reinterpret_cast<BatchNormParameter *>(op_parameter_);
  param->channel_ = input_shapes[n_dim - 1];
  param->unit_ = 1;
  for (size_t i = 0; i < n_dim - 1; i++) {
    param->unit_ *= input_shapes[i];
  }
}

int BatchnormCPUKernel::InitConstTensor() {
  mean_ = malloc(in_tensors_[1]->Size());
  variance_ = malloc(in_tensors_[2]->Size());
  if (mean_ == nullptr || variance_ == nullptr) {
    MS_LOG(ERROR) << "Memory allocation failed";
    FreeMeanAndVariance();
    return RET_ERROR;
  }
  memcpy(mean_, in_tensors_[1]->Data(), in_tensors_[1]->Size());
  memcpy(variance_, in_tensors_[2]->Data(), in_tensors_[2]->Size());
  return RET_OK;
}

int BatchnormCPUKernel::Run() {
  auto ret = Prepare();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail! Ret error code: " << ret;
    return ret;
  }
  ret = LiteBackendParallelLaunch(BatchNormRun, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "BatchnormRun error error_code[" << ret << "]";
  }
  return ret;
}

int BatchnormCPUKernel::DoExecute(int task_id) {
  auto param = reinterpret_cast<BatchNormParameter *>(op_parameter_);
  BatchNormFp32(in_tensors_.at(0)->Data(), mean_, variance_, param, task_id, out_tensors_.at(0)->Data());
  return mindspore::lite::RET_OK;
}

int BatchNormRun(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto kernel = reinterpret_cast<BatchnormCPUKernel *>(cdata);
  auto ret = kernel->DoExecute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "BatchnormRun error task_id[" << task_id << "] error_code[" << ret << "]";
  }
  return ret;
}

kernel::LiteKernel *CpuBatchnormKernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                              const std::vector<lite::tensor::Tensor *> &outputs,
                                              OpParameter *opParameter, const lite::Context *ctx,
                                              const kernel::KernelKey &desc,
                                              const mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(opParameter != nullptr);
  auto *kernel = new (std::nothrow) BatchnormCPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new BatchNormCPUKernel fail!";
    return nullptr;
  }
  auto ret = kernel->Init();
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Init kernel failed, name: " << opParameter->name_ << ", type: "
                  << schema::EnumNamePrimitiveType(static_cast<schema::PrimitiveType>(opParameter->type_));
    delete kernel;
    return nullptr;
  }
  return kernel;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_BatchNorm, CpuBatchnormKernelCreator)
}  // namespace mindspore::kernel
