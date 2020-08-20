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

#include "src/runtime/kernel/arm/int8/batchnorm_int8.h"
#include <math.h>
#include "schema/model_generated.h"
#include "src/kernel_registry.h"
#include "include/errorcode.h"
#include "src/runtime/runtime_api.h"
#include "src/runtime/kernel/arm/nnacl/batchnorm_parameter.h"

using mindspore::kernel::KERNEL_ARCH::kCPU;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_BatchNorm;
using mindspore::schema::PrimitiveType_FusedBatchNorm;

namespace mindspore::kernel {
BatchnormInt8CPUKernel::~BatchnormInt8CPUKernel() {
  if (alpha_addr_ != nullptr) {
    free(alpha_addr_);
    alpha_addr_ = nullptr;
  }
  if (beta_addr_ != nullptr) {
    free(beta_addr_);
    beta_addr_ = nullptr;
  }
}

int BatchnormInt8CPUKernel::InitConstTensor() {
  auto input = in_tensors_[0];
  auto mean = in_tensors_[1];
  auto variance = in_tensors_[2];
  auto output = out_tensors_[0];

  auto mean_ptr = reinterpret_cast<int8_t *>(mean->Data());
  auto var_ptr = reinterpret_cast<int8_t *>(variance->Data());
  alpha_addr_ = reinterpret_cast<float *>(malloc(mean->ElementsNum() * sizeof(float)));
  if (alpha_addr_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  beta_addr_ = reinterpret_cast<float *>(malloc(variance->ElementsNum() * sizeof(float)));
  if (beta_addr_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  // compute alpha, beta;
  // 0. tmp = (S4 * Sqrt(e + S3 * (q3 - Z3)));
  // 1. A = S1 / tmp;
  // 2. B = Z4 - (A1 * Z1) -((S2 * (q2 - Z2)) / tmp;
  auto eps = batchnorm_param_->epsilon_;
  auto zp_in = input->GetQuantParams().front().zeroPoint;
  auto zp_mean = mean->GetQuantParams().front().zeroPoint;
  auto zp_var = variance->GetQuantParams().front().zeroPoint;
  auto zp_out = output->GetQuantParams().front().zeroPoint;
  auto s_in = input->GetQuantParams().front().scale;
  auto s_mean = mean->GetQuantParams().front().scale;
  auto s_var = variance->GetQuantParams().front().scale;
  auto s_out = output->GetQuantParams().front().scale;

  for (int i = 0; i < batchnorm_param_->channel_; ++i) {
    float tmp = s_out * sqrt(eps + s_var * (var_ptr[i] - zp_var));
    float tmp_a = s_in / tmp;
    float tmp_b = zp_out - tmp_a * zp_in - (s_mean * (mean_ptr[i] - zp_mean)) / tmp;
    alpha_addr_[i] = tmp_a;
    beta_addr_[i] = tmp_b;
  }
  return RET_OK;
}

int BatchnormInt8CPUKernel::InitFusedConstTensor() {
  auto input = in_tensors_[0];
  auto scale = in_tensors_[1];
  auto offset = in_tensors_[2];
  auto mean = in_tensors_[3];
  auto variance = in_tensors_[4];
  auto output = out_tensors_[0];

  auto scale_ptr = reinterpret_cast<int8_t *>(scale->Data());
  auto offset_ptr = reinterpret_cast<int8_t *>(offset->Data());
  auto mean_ptr = reinterpret_cast<int8_t *>(mean->Data());
  auto var_ptr = reinterpret_cast<int8_t *>(variance->Data());

  alpha_addr_ = reinterpret_cast<float *>(malloc(mean->ElementsNum() * sizeof(float)));
  if (alpha_addr_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  beta_addr_ = reinterpret_cast<float *>(malloc(variance->ElementsNum() * sizeof(float)));
  if (beta_addr_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  // compute alpha, beta;
  // 0. tmp = (S6 * Sqrt(e + S5 * (q5 - Z5)));
  // 1. A = S1 * S2 * (q2 - Z2) / tmp;
  // 2. B = Z6 - (A1 * Z1) -((S3 * (q3 - Z3)) / S6 - S2 * S4 * (q2 - Z4) * (q4 - z4) / tmp;
  auto eps = batchnorm_param_->epsilon_;
  auto zp_in = input->GetQuantParams().front().zeroPoint;
  auto zp_scale = scale->GetQuantParams().front().zeroPoint;
  auto zp_offset = offset->GetQuantParams().front().zeroPoint;
  auto zp_mean = mean->GetQuantParams().front().zeroPoint;
  auto zp_var = variance->GetQuantParams().front().zeroPoint;
  auto zp_out = output->GetQuantParams().front().zeroPoint;
  auto s_in = input->GetQuantParams().front().scale;
  auto s_scale = scale->GetQuantParams().front().scale;
  auto s_offset = offset->GetQuantParams().front().scale;
  auto s_mean = mean->GetQuantParams().front().scale;
  auto s_var = variance->GetQuantParams().front().scale;
  auto s_out = output->GetQuantParams().front().scale;

  float mul_12 = s_in * s_scale;
  float mul_24 = s_scale * s_mean;
  float div_36 = s_offset / s_out;
  for (int i = 0; i < batchnorm_param_->channel_; ++i) {
    float tmp = s_out * sqrt(eps + s_var * (var_ptr[i] - zp_var));
    float tmp_a = (mul_12 * (scale_ptr[i] - zp_scale)) / tmp;
    float tmp_b = zp_out + div_36 * (offset_ptr[i] - zp_offset) - tmp_a * zp_in -
                  (mul_24 * (scale_ptr[i] - zp_scale) * (mean_ptr[i] - zp_mean)) / tmp;
    alpha_addr_[i] = tmp_a;
    beta_addr_[i] = tmp_b;
  }
  return RET_OK;
}

int BatchnormInt8CPUKernel::Init() {
  auto input_shapes = in_tensors_[0]->shape();
  auto n_dim = input_shapes.size();
  batchnorm_param_->channel_ = input_shapes[n_dim - 1];
  batchnorm_param_->units_ = 1;
  for (int i = 0; i < n_dim - 1; i++) {
    batchnorm_param_->units_ *= input_shapes[i];
  }
  batchnorm_param_->op_parameter_.thread_num_ =
    MSMIN(batchnorm_param_->op_parameter_.thread_num_, batchnorm_param_->channel_);
  batchnorm_param_->unit_ = UP_DIV(batchnorm_param_->units_, batchnorm_param_->op_parameter_.thread_num_);
  if (batchnorm_param_->fused_) {
    auto ret = InitFusedConstTensor();
    if (ret != 0) {
      MS_LOG(ERROR) << "FusedBatchnorm int8 InitFusedConstTensor failed.";
      return RET_ERROR;
    }
  } else {
    auto ret = InitConstTensor();
    if (ret != 0) {
      MS_LOG(ERROR) << "Batchnorm int8 InitConstTensor failed.";
      return RET_ERROR;
    }
  }

  return RET_OK;
}

int BatchnormInt8CPUKernel::ReSize() {
  auto input_shapes = in_tensors_[0]->shape();
  batchnorm_param_->unit_ = 1;
  for (int i = 0; i < input_shapes.size() - 1; i++) {
    batchnorm_param_->unit_ *= input_shapes[i];
  }
  return RET_OK;
}

int BatchnormInt8CPUKernel::DoExecute(int task_id) {
  BatchNormInt8(out_addr_, in_addr_, alpha_addr_, beta_addr_, task_id, batchnorm_param_);
  return RET_OK;
}

int BatchNormInt8Run(int task_id, LiteParallelGroupEnv *penv, void *cdata) {
  auto g_kernel = reinterpret_cast<BatchnormInt8CPUKernel *>(cdata);
  auto ret = g_kernel->DoExecute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "BatchnormRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int BatchnormInt8CPUKernel::Run() {
  auto prepare_ret = Prepare();
  if (prepare_ret != RET_OK) {
    MS_LOG(ERROR) << "Prepare fail! Ret error code: " << prepare_ret;
    return prepare_ret;
  }
  in_addr_ = reinterpret_cast<int8_t *>(in_tensors_.at(0)->Data());
  out_addr_ = reinterpret_cast<int8_t *>(out_tensors_.at(0)->Data());

  int ret = LiteBackendParallelLaunch(BatchNormInt8Run, this, batchnorm_param_->op_parameter_.thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "BatchnormRun error error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

kernel::LiteKernel *CpuBatchnormInt8KernelCreator(const std::vector<lite::tensor::Tensor *> &inputs,
                                                  const std::vector<lite::tensor::Tensor *> &outputs,
                                                  OpParameter *opParameter, const lite::Context *ctx,
                                                  const kernel::KernelKey &desc,
                                                  const mindspore::lite::PrimitiveC *primitive) {
  MS_ASSERT(opParameter != nullptr);
  MS_ASSERT(desc.type == schema::PrimitiveType_BatchNorm);
  auto *kernel = new (std::nothrow) BatchnormInt8CPUKernel(opParameter, inputs, outputs, ctx, primitive);
  if (kernel == nullptr) {
    MS_LOG(ERROR) << "new BatchnormInt8CPUKernel fail!";
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

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_BatchNorm, CpuBatchnormInt8KernelCreator)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_FusedBatchNorm, CpuBatchnormInt8KernelCreator)
}  // namespace mindspore::kernel
