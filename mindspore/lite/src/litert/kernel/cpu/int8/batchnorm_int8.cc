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

#include "src/litert/kernel/cpu/int8/batchnorm_int8.h"
#include <cmath>
#include "schema/model_generated.h"
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"
#include "nnacl/batchnorm_parameter.h"

using mindspore::kernel::KERNEL_ARCH;
using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_BatchNorm;
using mindspore::schema::PrimitiveType_FusedBatchNorm;
namespace {
constexpr int kNumInput0 = 0;
constexpr int kNumInput1 = 1;
constexpr int kNumInput2 = 2;
constexpr int kNumInput3 = 3;
constexpr int kNumInput4 = 4;
}  // namespace
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
  auto input = in_tensors_.at(kNumInput0);
  auto mean = in_tensors_.at(kNumInput1);
  auto variance = in_tensors_.at(kNumInput2);
  auto output = out_tensors_.at(0);
  MS_CHECK_FALSE_MSG(mean->shape().empty() || variance->shape().empty(), RET_ERROR, "Invalid input tensors.");

  auto mean_ptr = reinterpret_cast<int8_t *>(mean->MutableData());
  CHECK_NULL_RETURN(mean_ptr);
  auto var_ptr = reinterpret_cast<int8_t *>(variance->MutableData());
  CHECK_NULL_RETURN(var_ptr);
  MS_CHECK_GT(mean->ElementsNum(), 0, RET_ERROR);
  CHECK_LESS_RETURN(MAX_MALLOC_SIZE, static_cast<size_t>(mean->ElementsNum()) * sizeof(float));
  alpha_addr_ = reinterpret_cast<float *>(malloc(static_cast<size_t>(mean->ElementsNum()) * sizeof(float)));
  if (alpha_addr_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  MS_CHECK_GT(variance->ElementsNum(), 0, RET_ERROR);
  CHECK_LESS_RETURN(MAX_MALLOC_SIZE, static_cast<size_t>(variance->ElementsNum()) * sizeof(float));
  beta_addr_ = reinterpret_cast<float *>(malloc(static_cast<size_t>(variance->ElementsNum()) * sizeof(float)));
  if (beta_addr_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  // compute alpha, beta;
  auto eps = batchnorm_param_->epsilon_;
  CHECK_LESS_RETURN(input->quant_params().size(), 1);
  CHECK_LESS_RETURN(mean->quant_params().size(), 1);
  CHECK_LESS_RETURN(variance->quant_params().size(), 1);
  CHECK_LESS_RETURN(output->quant_params().size(), 1);
  auto zp_in = input->quant_params().front().zeroPoint;
  auto zp_mean = mean->quant_params().front().zeroPoint;
  auto zp_var = variance->quant_params().front().zeroPoint;
  auto zp_out = output->quant_params().front().zeroPoint;
  auto s_in = input->quant_params().front().scale;
  auto s_mean = mean->quant_params().front().scale;
  auto s_var = variance->quant_params().front().scale;
  auto s_out = output->quant_params().front().scale;

  if (batchnorm_param_->channel_ > variance->ElementsNum()) {
    MS_LOG(ERROR) << "Buffer overflow error.";
    return RET_ERROR;
  }
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
  CHECK_LESS_RETURN(in_tensors_.size(), DIMENSION_5D);
  auto input = in_tensors_.at(kNumInput0);
  auto scale = in_tensors_.at(kNumInput1);
  auto offset = in_tensors_.at(kNumInput2);
  CHECK_NULL_RETURN(in_tensors_.at(kNumInput3));
  auto mean = in_tensors_.at(kNumInput3);
  CHECK_NULL_RETURN(in_tensors_.at(kNumInput4));
  auto variance = in_tensors_.at(kNumInput4);
  auto output = out_tensors_.at(0);

  auto scale_ptr = reinterpret_cast<int8_t *>(scale->MutableData());
  CHECK_NULL_RETURN(scale_ptr);
  auto offset_ptr = reinterpret_cast<int8_t *>(offset->MutableData());
  CHECK_NULL_RETURN(offset_ptr);
  auto mean_ptr = reinterpret_cast<int8_t *>(mean->MutableData());
  CHECK_NULL_RETURN(mean_ptr);
  auto var_ptr = reinterpret_cast<int8_t *>(variance->MutableData());
  CHECK_NULL_RETURN(var_ptr);

  MS_CHECK_GT(mean->ElementsNum(), 0, RET_ERROR);
  CHECK_LESS_RETURN(MAX_MALLOC_SIZE, static_cast<size_t>(mean->ElementsNum()) * sizeof(float));
  alpha_addr_ = reinterpret_cast<float *>(malloc(static_cast<size_t>(mean->ElementsNum()) * sizeof(float)));
  if (alpha_addr_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  MS_CHECK_GT(variance->ElementsNum(), 0, RET_ERROR);
  CHECK_LESS_RETURN(MAX_MALLOC_SIZE, static_cast<size_t>(variance->ElementsNum()) * sizeof(float));
  beta_addr_ = reinterpret_cast<float *>(malloc(static_cast<size_t>(variance->ElementsNum()) * sizeof(float)));
  if (beta_addr_ == nullptr) {
    MS_LOG(ERROR) << "Malloc buffer failed.";
    return RET_ERROR;
  }
  // compute alpha, beta;
  auto eps = batchnorm_param_->epsilon_;
  CHECK_LESS_RETURN(input->quant_params().size(), 1);
  CHECK_LESS_RETURN(scale->quant_params().size(), 1);
  CHECK_LESS_RETURN(offset->quant_params().size(), 1);
  CHECK_LESS_RETURN(mean->quant_params().size(), 1);
  CHECK_LESS_RETURN(variance->quant_params().size(), 1);
  CHECK_LESS_RETURN(output->quant_params().size(), 1);
  auto zp_in = input->quant_params().front().zeroPoint;
  auto zp_scale = scale->quant_params().front().zeroPoint;
  auto zp_offset = offset->quant_params().front().zeroPoint;
  auto zp_mean = mean->quant_params().front().zeroPoint;
  auto zp_var = variance->quant_params().front().zeroPoint;
  auto zp_out = output->quant_params().front().zeroPoint;
  auto s_in = input->quant_params().front().scale;
  auto s_scale = scale->quant_params().front().scale;
  auto s_offset = offset->quant_params().front().scale;
  auto s_mean = mean->quant_params().front().scale;
  auto s_var = variance->quant_params().front().scale;
  auto s_out = output->quant_params().front().scale;

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

int BatchnormInt8CPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), DIMENSION_3D);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(in_tensors_[kNumInput0]);
  CHECK_NULL_RETURN(in_tensors_[kNumInput1]);
  CHECK_NULL_RETURN(in_tensors_[kNumInput2]);
  CHECK_NULL_RETURN(out_tensors_[0]);
  if (in_tensors_[kNumInput0]->data_type() != mindspore::kNumberTypeInt8 ||
      in_tensors_[kNumInput1]->data_type() != mindspore::kNumberTypeInt8 ||
      in_tensors_[kNumInput2]->data_type() != mindspore::kNumberTypeInt8 ||
      out_tensors_[0]->data_type() != mindspore::kNumberTypeInt8) {
    MS_LOG(ERROR) << "Datatype error, input0 data_type is " << in_tensors_[kNumInput0]->data_type()
                  << ", input1 data_type is " << in_tensors_[kNumInput1]->data_type() << ", input2 data_type is "
                  << in_tensors_[kNumInput2]->data_type() << ", output data_type is " << out_tensors_[0]->data_type();
    return RET_ERROR;
  }
  CHECK_NULL_RETURN(batchnorm_param_);
  auto input_shapes = in_tensors_.at(0)->shape();
  auto n_dim = input_shapes.size();
  CHECK_LESS_RETURN(n_dim, 1);
  batchnorm_param_->channel_ = input_shapes[n_dim - 1];
  batchnorm_param_->units_ = 1;
  for (size_t i = 0; i < n_dim - 1; i++) {
    batchnorm_param_->units_ *= input_shapes[i];
  }
  batchnorm_param_->op_parameter_.thread_num_ =
    MSMIN(batchnorm_param_->op_parameter_.thread_num_, batchnorm_param_->channel_);
  if (batchnorm_param_->op_parameter_.thread_num_ == 0) {
    MS_LOG(ERROR) << "div zero";
    return RET_ERROR;
  }
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
  auto input_shapes = in_tensors_.at(0)->shape();
  batchnorm_param_->unit_ = 1;
  for (size_t i = 0; i < input_shapes.size() - 1; i++) {
    batchnorm_param_->unit_ *= input_shapes[i];
  }
  return RET_OK;
}

int BatchnormInt8CPUKernel::DoExecute(int task_id) {
  BatchNormInt8(out_addr_, in_addr_, alpha_addr_, beta_addr_, task_id, batchnorm_param_);
  return RET_OK;
}

int BatchNormInt8Run(void *cdata, int task_id, float, float) {
  CHECK_NULL_RETURN(cdata);
  auto g_kernel = reinterpret_cast<BatchnormInt8CPUKernel *>(cdata);
  auto ret = g_kernel->DoExecute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "BatchnormRun error task_id[" << task_id << "] error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

int BatchnormInt8CPUKernel::Run() {
  in_addr_ = reinterpret_cast<int8_t *>(in_tensors_.at(0)->MutableData());
  CHECK_NULL_RETURN(in_addr_);
  out_addr_ = reinterpret_cast<int8_t *>(out_tensors_.at(0)->MutableData());
  CHECK_NULL_RETURN(out_addr_);

  auto ret = ParallelLaunch(this->ms_context_, BatchNormInt8Run, this, batchnorm_param_->op_parameter_.thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "BatchnormRun error error_code[" << ret << "]";
    return ret;
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_BatchNorm, LiteKernelCreator<BatchnormInt8CPUKernel>)
REG_KERNEL(kCPU, kNumberTypeInt8, PrimitiveType_FusedBatchNorm, LiteKernelCreator<BatchnormInt8CPUKernel>)
}  // namespace mindspore::kernel
