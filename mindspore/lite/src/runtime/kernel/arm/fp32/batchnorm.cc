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
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_BatchNorm;

namespace mindspore::kernel {
BatchnormCPUKernel::~BatchnormCPUKernel() {
  if (mean_addr_ != nullptr) {
    free(mean_addr_);
    mean_addr_ = nullptr;
  }
  if (var_addr_ != nullptr) {
    free(var_addr_);
    var_addr_ = nullptr;
  }
}

int BatchnormCPUKernel::InitConstTensor() {
  auto mean = in_tensors_[1];
  mean_addr_ = reinterpret_cast<float *>(malloc(mean->ElementsNum() * sizeof(float)));
  if (mean_addr_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  memcpy(mean_addr_, mean->Data(), mean->ElementsNum() * sizeof(float));

  auto variance = in_tensors_[2];
  var_addr_ = reinterpret_cast<float *>(malloc(variance->ElementsNum() * sizeof(float)));
  if (var_addr_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  memcpy(var_addr_, variance->Data(), variance->ElementsNum() * sizeof(float));
  return RET_OK;
}

int BatchnormCPUKernel::Init() {
  if (context_->infer_shape_interrupt_ && !context_->running_) {
    set_need_reinit();
    return RET_OK;
  }

  auto input_shapes = in_tensors_[0]->shape();
  auto n_dim = input_shapes.size();
  batchnorm_param_->channel_ = input_shapes[n_dim - 1];
  batchnorm_param_->unit_ = 1;
  for (int i = 0; i < n_dim - 1; i++) {
    batchnorm_param_->unit_ *= input_shapes[i];
  }
  batchnorm_param_->op_parameter_.thread_num_ =
    MSMIN(batchnorm_param_->op_parameter_.thread_num_, batchnorm_param_->channel_);

  auto ret = InitConstTensor();
  if (ret != 0) {
    MS_LOG(ERROR) << "Batchnorm fp32 InitConstTensor failed.";
    return RET_ERROR;
  }
  return RET_OK;
}

int BatchnormCPUKernel::ReSize() {
  auto input_shapes = in_tensors_[0]->shape();
  batchnorm_param_->unit_ = 1;
  for (int i = 0; i < input_shapes.size() - 1; i++) {
    batchnorm_param_->unit_ *= input_shapes[i];
  }
  return RET_OK;
}

int BatchnormCPUKernel::DoExecute(int task_id) {
  BatchNorm(out_addr_, in_addr_, mean_addr_, var_addr_, task_id, batchnorm_param_);
  return RET_OK;
}

int BatchNormRun(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto g_kernel = reinterpret_cast<BatchnormCPUKernel *>(cdata);
  auto ret = g_kernel->DoExecute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "BatchnormRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int BatchnormCPUKernel::Run() {
  auto prepare_ret = Prepare();
  if (prepare_ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail! Ret error code: " << prepare_ret;
    return prepare_ret;
  }
  in_addr_ = reinterpret_cast<float *>(in_tensors_.at(0)->Data());
  out_addr_ = reinterpret_cast<float *>(out_tensors_.at(0)->Data());

  int ret = LiteBackendParallelLaunch(BatchNormRun, this, batchnorm_param_->op_parameter_.thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "BatchnormRun error error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuBatchnormKernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                              const std::vector<lite::tensor::Tensor *> &outputs,
                                              OpParameter *opParameter, const lite::Context *ctx,
                                              const kernel::KernelKey &desc, const lite::Primitive *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_BatchNorm);
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
