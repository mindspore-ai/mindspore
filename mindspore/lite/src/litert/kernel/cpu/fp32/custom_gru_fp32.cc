#ifdef ENABLE_ARM64
/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#include "src/litert/kernel/cpu/fp32/custom_gru_fp32.h"
#include <algorithm>
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "src/litert/pack_weight_manager.h"
#include "nnacl/fp32/pack_fp32.h"
#include "nnacl/custom_gru_parameter.h"
#include "nnacl/fp32/custom_gru_fp32.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NOT_SUPPORT;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
CustomGruCPUKernel::~CustomGruCPUKernel() {
  if (weight_in_) {
    lite::PackWeightManager::GetInstance()->Free(weight_in_);
    weight_in_ = nullptr;
  }
  if (weight_hidden_) {
    lite::PackWeightManager::GetInstance()->Free(weight_hidden_);
    weight_hidden_ = nullptr;
  }
  if (bias_in_) {
    free(bias_in_);
    bias_in_ = nullptr;
    bias_hidden_ = nullptr;
  }
  if (in_tensors_[SIXTH_INPUT]->IsConst() && init_h_) {
    free(init_h_);
    init_h_ = nullptr;
  }
}

int CustomGruCPUKernel::Prepare() {
  CHECK_LESS_RETURN(in_tensors_.size(), C6NUM);
  CHECK_LESS_RETURN(out_tensors_.size(), 1);
  if (in_tensors_[FIRST_INPUT]->IsConst()) {
    MS_LOG(ERROR) << "Built-in CustomGru first-input must be a variable." << name_;
    return RET_NOT_SUPPORT;
  }
  for (size_t i = 1; i < C5NUM; ++i) {
    if (!in_tensors_[i]->IsConst()) {
      MS_LOG(ERROR) << "Built-in CustomGru only support first-input and last-input is variable." << name_;
      return RET_NOT_SUPPORT;
    }
  }
  if (InitParamter() != RET_OK) {
    MS_LOG(ERROR) << "Init Built-in CustomGru Parameter failed." << name_;
    return RET_ERROR;
  }
  if (InitWeightAndBias() != RET_OK) {
    MS_LOG(ERROR) << "Init Built-in CustomGru Weight and bias failed." << name_;
    return RET_ERROR;
  }
  if (!InferShapeDone()) {
    return RET_OK;
  }
  return ReSize();
}

int CustomGruCPUKernel::InitParamter() {
  auto param = reinterpret_cast<CustomGruParameter *>(op_parameter_);
  thread_num_ = 1;
  auto weight_in_shape = in_tensors_[1]->shape();
  auto weight_hidden_shape = in_tensors_[C2NUM]->shape();
  if (weight_in_shape.size() != C2NUM || weight_hidden_shape.size() != C2NUM) {
    MS_LOG(ERROR) << "Built-in CustomGru's weight must be 2D." << name_;
    return RET_ERROR;
  }
  if (weight_in_shape[0] != weight_hidden_shape[0]) {
    MS_LOG(ERROR) << "Built-in CustomGru's weight-in and weight-hidden first-dim must be same." << name_;
    return RET_ERROR;
  }
  if (weight_hidden_shape[0] != weight_hidden_shape[1] * C3NUM) {
    MS_LOG(ERROR) << "Built-in CustomGru's weight-hidden first-dim must be 3 * second-dim." << name_;
    return RET_ERROR;
  }
  auto bias_in_shape = in_tensors_[C3NUM]->shape();
  auto bias_hidden_shape = in_tensors_[C4NUM]->shape();
  if (bias_in_shape.size() != 1) {
    MS_LOG(ERROR) << "Built-in CustomGru's bias must be 1D." << name_;
    return RET_ERROR;
  }
  if (bias_in_shape != bias_hidden_shape) {
    MS_LOG(ERROR) << "Built-in CustomGru's bias-in and bias-hidden must have same shape." << name_;
    return RET_ERROR;
  }
  if (bias_in_shape.back() != weight_in_shape.front()) {
    MS_LOG(ERROR) << "Built-in CustomGru's bias-in shape don't match with the first-dim of weight." << name_;
    return RET_ERROR;
  }
  if (bias_in_shape.front() % C3NUM != 0) {
    MS_LOG(ERROR) << "The first-dim of CustomGru's weight must be 3 * hidden.";
    return RET_ERROR;
  }
  param->input_size = weight_in_shape.back();
  param->hidden_size = bias_in_shape.front() / C3NUM;
  return RET_OK;
}

int CustomGruCPUKernel::InitWeightAndBias() {
  auto weight_shape = in_tensors_[1]->shape();
  auto hidden_size = weight_shape[0] / C3NUM;
  auto col_align = UP_ROUND(hidden_size, col_tile_);
  auto weight_in_pack_size = static_cast<size_t>(col_align * weight_shape[1]) * sizeof(float);
  bool is_packed = false;
  weight_in_ = lite::PackWeightManager::GetInstance()->GetPackData(
    in_tensors_[SECOND_INPUT]->data(), static_cast<size_t>(weight_in_pack_size * C3NUM), &is_packed);
  MS_CHECK_TRUE_MSG(weight_in_ != nullptr, lite::RET_NULL_PTR, "malloc for packing weight-in failed.");
  if (!is_packed) {
    auto weight_in_src = static_cast<const float *>(in_tensors_[SECOND_INPUT]->data());
    for (int i = 0; i < C3NUM; ++i) {
      RowMajor2Col8MajorParallel(weight_in_src + i * hidden_size * weight_shape[1],
                                 static_cast<float *>(weight_in_) + i * col_align * weight_shape[1], hidden_size,
                                 weight_shape[1], 0, hidden_size);
    }
  }
  auto weight_hidden_pack_size = static_cast<size_t>(col_align * hidden_size) * sizeof(float);
  is_packed = false;
  weight_hidden_ = lite::PackWeightManager::GetInstance()->GetPackData(
    in_tensors_[THIRD_INPUT]->data(), static_cast<size_t>(weight_hidden_pack_size * C3NUM), &is_packed);
  MS_CHECK_TRUE_MSG(weight_hidden_ != nullptr, lite::RET_NULL_PTR, "malloc for packing weight-hidden failed.");
  if (!is_packed) {
    auto weight_hidden_src = static_cast<const float *>(in_tensors_[THIRD_INPUT]->data());
    for (int i = 0; i < C3NUM; ++i) {
      RowMajor2Col8MajorParallel(weight_hidden_src + i * hidden_size * weight_shape[1],
                                 static_cast<float *>(weight_hidden_) + i * col_align * weight_shape[1], hidden_size,
                                 hidden_size, 0, hidden_size);
    }
  }
  auto bias_pack_size = static_cast<size_t>(col_align) * sizeof(float);
  auto bias = reinterpret_cast<float *>(malloc(bias_pack_size * C6NUM));
  if (bias == nullptr) {
    MS_LOG(ERROR) << "malloc for packing bias failed.";
    return lite::RET_NULL_PTR;
  }
  (void)memset(bias, 0, bias_pack_size * C6NUM);
  bias_in_ = bias;
  bias_hidden_ = bias + col_align * C3NUM;
  auto bias_in_src = static_cast<const float *>(in_tensors_[FOURTH_INPUT]->data());
  for (int i = 0; i < C3NUM; ++i) {
    (void)memcpy(bias + i * col_align, bias_in_src + i * hidden_size, hidden_size * sizeof(float));
  }
  auto bias_hidden_src = static_cast<const float *>(in_tensors_[FIFTH_INPUT]->data());
  for (int i = 0; i < C3NUM; ++i) {
    (void)memcpy(bias + (C3NUM + i) * col_align, bias_hidden_src + i * hidden_size, hidden_size * sizeof(float));
  }
  if (in_tensors_[SIXTH_INPUT]->IsConst()) {
    init_h_ = malloc(in_tensors_[SIXTH_INPUT]->Size());
    MS_CHECK_TRUE_MSG(init_h_ != nullptr, lite::RET_NULL_PTR, "malloc for init-h failed.");
    (void)memcpy(init_h_, in_tensors_[SIXTH_INPUT]->data(), in_tensors_[SIXTH_INPUT]->Size());
  }
  return RET_OK;
}

int CustomGruCPUKernel::ReSize() {
  auto in_shape = in_tensors_.front()->shape();
  if (in_shape.size() != C3NUM) {
    MS_LOG(ERROR) << "Built-in CustomGru's first-input must be 3D." << name_;
    return RET_ERROR;
  }
  auto param = reinterpret_cast<CustomGruParameter *>(op_parameter_);
  param->num_step = in_shape[0];
  param->batch_size = in_shape[1];
  if (in_shape.back() != param->input_size) {
    MS_LOG(ERROR) << "Built-in CustomGru's fisrt-input don't match its weight." << name_;
    return RET_ERROR;
  }
  return RET_OK;
}

void CustomGruCPUKernel::MallocRunBuffer(size_t data_type_size) {
  if (run_buffer_ != nullptr) {
    return;
  }
  auto param = reinterpret_cast<CustomGruParameter *>(op_parameter_);
  auto row_align = UP_ROUND(param->batch_size, row_tile_);
  auto run_buffer_size =
    (row_align * (param->input_size + param->hidden_size) + param->batch_size * param->hidden_size * C6NUM) *
    data_type_size;
  if (ms_context_->allocator != nullptr) {
    run_buffer_ = ms_context_->allocator->Malloc(run_buffer_size);
  } else {
    run_buffer_ = malloc(run_buffer_size);
  }
}

int CustomGruCPUKernel::Run() {
  auto input = reinterpret_cast<float *>(in_tensors_[FIRST_INPUT]->data());
  if (input == nullptr) {
    MS_LOG(ERROR) << "Built-in CustomGru's fisrt-input is nullptr." << name_;
    return lite::RET_NULL_PTR;
  }
  if (!in_tensors_[SIXTH_INPUT]->IsConst()) {
    init_h_ = in_tensors_[SIXTH_INPUT]->data();
  }
  if (init_h_ == nullptr) {
    MS_LOG(ERROR) << "Built-in CustomGru's six-input is nullptr." << name_;
    return lite::RET_NULL_PTR;
  }
  auto output = reinterpret_cast<float *>(out_tensors_.front()->data());
  if (output == nullptr) {
    MS_LOG(ERROR) << "Built-in CustomGru's output is nullptr." << name_;
    return lite::RET_NULL_PTR;
  }
  MallocRunBuffer(sizeof(float));
  if (run_buffer_ == nullptr) {
    MS_LOG(ERROR) << "malloc running buffer failed." << name_;
    return lite::RET_NULL_PTR;
  }
  auto param = reinterpret_cast<CustomGruParameter *>(op_parameter_);
  auto row_align = UP_ROUND(param->batch_size, row_tile_);
  auto run_buffer = reinterpret_cast<float *>(run_buffer_);
  float *buffer[C4NUM] = {
    run_buffer, run_buffer + row_align * param->input_size,
    run_buffer + row_align * param->input_size + param->batch_size * param->hidden_size * C3NUM,
    run_buffer + row_align * (param->input_size + param->hidden_size) + param->batch_size * param->hidden_size * C3NUM};
  CustomGru(output, input, static_cast<float *>(weight_in_), static_cast<float *>(weight_hidden_),
            static_cast<float *>(bias_in_), static_cast<float *>(bias_hidden_), static_cast<float *>(init_h_), buffer,
            param);
  if (ms_context_->allocator != nullptr) {
    ms_context_->allocator->Free(run_buffer_);
  } else {
    free(run_buffer_);
  }
  run_buffer_ = nullptr;
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat32, PrimType_Inner_CustomGru, LiteKernelCreator<CustomGruCPUKernel>)
}  // namespace mindspore::kernel
#endif
