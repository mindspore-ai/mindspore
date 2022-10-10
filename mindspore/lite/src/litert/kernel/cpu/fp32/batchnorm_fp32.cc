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

#include "src/litert/kernel/cpu/fp32/batchnorm_fp32.h"
#include "src/litert/kernel_registry.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NULL_PTR;
using mindspore::lite::RET_OK;
using mindspore::schema::PrimitiveType_BatchNorm;
namespace {
constexpr int kNumInput2 = 2;
constexpr int kMeanInput = 1;
constexpr int kVarInput = 2;
}  // namespace
namespace mindspore::kernel {
int BatchnormCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), DIMENSION_3D);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  CHECK_NULL_RETURN(in_tensors_[0]);
  CHECK_NULL_RETURN(in_tensors_[kMeanInput]);
  CHECK_NULL_RETURN(in_tensors_[kVarInput]);
  if (in_tensors_[0]->data_type() != kNumberTypeFloat16 && in_tensors_[0]->data_type() != kNumberTypeFloat32) {
    MS_LOG(ERROR) << "Invalid in_tensor[0] data_type: " << in_tensors_[0]->data_type();
    return RET_ERROR;
  }
  if (in_tensors_[kMeanInput]->data_type() != kNumberTypeFloat16 &&
      in_tensors_[kMeanInput]->data_type() != kNumberTypeFloat32) {
    MS_LOG(ERROR) << "Invalid in_tensor[1] data_type: " << in_tensors_[kMeanInput]->data_type();
    return RET_ERROR;
  }
  if (in_tensors_[kVarInput]->data_type() != kNumberTypeFloat16 &&
      in_tensors_[kVarInput]->data_type() != kNumberTypeFloat32) {
    MS_LOG(ERROR) << "Invalid in_tensor[2] data_type: " << in_tensors_[kVarInput]->data_type();
    return RET_ERROR;
  }

  CHECK_NULL_RETURN(out_tensors_[0]);
  if (out_tensors_[0]->data_type() != kNumberTypeFloat16 && out_tensors_[0]->data_type() != kNumberTypeFloat32) {
    MS_LOG(ERROR) << "Invalid out_tensors[0] data_type: " << out_tensors_[0]->data_type();
    return RET_ERROR;
  }
  CHECK_NULL_RETURN(op_parameter_);
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

int BatchnormCPUKernel::FillParam() {
  auto input_shapes = in_tensors_.at(0)->shape();
  auto in_n_dim = input_shapes.size();
  CHECK_LESS_RETURN(in_n_dim, 1);
  auto in_channel = input_shapes[in_n_dim - 1];

  auto mean_shapes = in_tensors_.at(kMeanInput)->shape();
  auto mean_n_dim = mean_shapes.size();
  CHECK_LESS_RETURN(mean_n_dim, 1);
  auto mean_channel = mean_shapes[mean_n_dim - 1];
  CHECK_NOT_EQUAL_RETURN(in_channel, mean_channel);

  auto var_shapes = in_tensors_.at(kVarInput)->shape();
  auto var_n_dim = var_shapes.size();
  CHECK_LESS_RETURN(var_n_dim, 1);
  auto var_channel = var_shapes[var_n_dim - 1];
  CHECK_NOT_EQUAL_RETURN(in_channel, var_channel);

  auto param = reinterpret_cast<BatchNormParameter *>(op_parameter_);
  param->channel_ = input_shapes[in_n_dim - 1];
  param->unit_ = 1;
  for (size_t i = 0; i < in_n_dim - 1; i++) {
    MS_CHECK_FALSE_MSG(INT_MUL_OVERFLOW(param->unit_, input_shapes[i]), RET_ERROR, "mul overflow.");
    param->unit_ *= input_shapes[i];
  }
  if (default_momentum_ < 0.0f) {
    default_momentum_ = param->momentum_;
  }
  return RET_OK;
}

int BatchnormCPUKernel::InitConstTensor() {
  CHECK_LESS_RETURN(MAX_MALLOC_SIZE, in_tensors_.at(1)->Size());
  CHECK_LESS_RETURN(MAX_MALLOC_SIZE, in_tensors_.at(kNumInput2)->Size());
  mean_ = malloc(in_tensors_.at(kMeanInput)->Size());
  variance_ = malloc(in_tensors_.at(kVarInput)->Size());
  if (mean_ == nullptr || variance_ == nullptr) {
    MS_LOG(ERROR) << "Memory allocation failed";
    FreeMeanAndVariance();
    return RET_ERROR;
  }
  auto in_tensor_mean_data = in_tensors_.at(kMeanInput)->MutableData();
  auto in_tensor_var_data = in_tensors_.at(kVarInput)->MutableData();
  if (in_tensor_mean_data == nullptr || in_tensor_var_data == nullptr) {
    FreeMeanAndVariance();
    return RET_ERROR;
  }
  (void)memcpy(mean_, in_tensor_mean_data, in_tensors_.at(kMeanInput)->Size());
  (void)memcpy(variance_, in_tensor_var_data, in_tensors_.at(kVarInput)->Size());
  return RET_OK;
}

int BatchnormCPUKernel::Run() {
  auto ret = ParallelLaunch(this->ms_context_, BatchNormRun, this, op_parameter_->thread_num_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "BatchnormRun error error_code[" << ret << "]";
  }
  return ret;
}

int BatchnormCPUKernel::DoExecute(int task_id) {
  auto param = reinterpret_cast<BatchNormParameter *>(op_parameter_);
  auto in_tensor_data = reinterpret_cast<float *>(in_tensors_.at(0)->data());
  CHECK_NULL_RETURN(in_tensor_data);
  auto out_tensor_data = reinterpret_cast<float *>(out_tensors_.at(0)->data());
  CHECK_NULL_RETURN(out_tensor_data);
  BatchNormFp32(in_tensor_data, reinterpret_cast<float *>(mean_), reinterpret_cast<float *>(variance_), param, task_id,
                out_tensor_data);
  return RET_OK;
}

int BatchNormRun(void *cdata, int task_id, float lhs_scale, float rhs_scale) {
  CHECK_NULL_RETURN(cdata);
  auto kernel = reinterpret_cast<BatchnormCPUKernel *>(cdata);
  auto ret = kernel->DoExecute(task_id);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "BatchnormRun error task_id[" << task_id << "] error_code[" << ret << "]";
  }
  return ret;
}

int BatchnormCPUKernel::set_momentum(float momentum) {
  auto param = reinterpret_cast<BatchNormParameter *>(op_parameter_);
  param->momentum_ = momentum;

  return RET_OK;
}

float BatchnormCPUKernel::get_momentum() {
  auto param = reinterpret_cast<BatchNormParameter *>(op_parameter_);
  return param->momentum_;
}

int BatchnormCPUKernel::RestoreDefaultMomentum() {
  auto ret = set_momentum(default_momentum_);
  if (ret != RET_OK) {
    MS_LOG(ERROR) << "Restore Momentum Error";
    return RET_ERROR;
  }
  return RET_OK;
}

int BatchnormCPUKernel::SetupVirtualBatch(int virtual_batch_multiplier, int param) {
  if ((virtual_batch_multiplier > 0)) {
    int momentum = (param < 0.0f) ? (this->get_momentum() / virtual_batch_multiplier) : param;
    return this->set_momentum(momentum);
  } else {
    return this->RestoreDefaultMomentum();
  }
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimitiveType_BatchNorm, LiteKernelCreator<BatchnormCPUKernel>)
}  // namespace mindspore::kernel
