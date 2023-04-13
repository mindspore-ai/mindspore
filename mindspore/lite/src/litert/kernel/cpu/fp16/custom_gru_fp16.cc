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
#include "src/litert/kernel/cpu/fp16/custom_gru_fp16.h"
#include <algorithm>
#include "src/litert/kernel_registry.h"
#include "include/errorcode.h"
#include "src/common/log_adapter.h"
#include "src/litert/pack_weight_manager.h"
#include "nnacl/fp16/pack_fp16.h"
#include "nnacl/custom_gru_parameter.h"
#include "nnacl/fp16/custom_gru_fp16.h"
#include "nnacl/fp16/matmul_fp16.h"

using mindspore::lite::KernelRegistrar;
using mindspore::lite::RET_ERROR;
using mindspore::lite::RET_NOT_SUPPORT;
using mindspore::lite::RET_OK;

namespace mindspore::kernel {
int CustomGruFp16CPUKernel::InitWeightAndBias() {
  auto weight_shape = in_tensors_[1]->shape();
  auto hidden_size = weight_shape[0] / C3NUM;
  auto col_align = UP_ROUND(hidden_size, col_tile_);
  auto weight_in_pack_size = static_cast<size_t>(col_align * weight_shape[1]) * sizeof(float16_t);
  bool is_packed = false;
  weight_in_ = lite::PackWeightManager::GetInstance()->GetPackData(
    in_tensors_[SECOND_INPUT]->data(), static_cast<size_t>(weight_in_pack_size * C3NUM), &is_packed);
  MS_CHECK_TRUE_MSG(weight_in_ != nullptr, lite::RET_NULL_PTR, "malloc for packing weight-in failed.");
  if (!is_packed) {
    auto weight_in_src = static_cast<const float16_t *>(in_tensors_[SECOND_INPUT]->data());
    for (int i = 0; i < C3NUM; ++i) {
      RowMajor2Col8MajorFp16(weight_in_src + i * hidden_size * weight_shape[1],
                             static_cast<float16_t *>(weight_in_) + i * col_align * weight_shape[1], hidden_size,
                             weight_shape[1], false);
    }
  }
  auto weight_hidden_pack_size = static_cast<size_t>(col_align * hidden_size) * sizeof(float16_t);
  is_packed = false;
  weight_hidden_ = lite::PackWeightManager::GetInstance()->GetPackData(
    in_tensors_[THIRD_INPUT]->data(), static_cast<size_t>(weight_hidden_pack_size * C3NUM), &is_packed);
  MS_CHECK_TRUE_MSG(weight_hidden_ != nullptr, lite::RET_NULL_PTR, "malloc for packing weight-hidden failed.");
  if (!is_packed) {
    auto weight_hidden_src = static_cast<const float16_t *>(in_tensors_[THIRD_INPUT]->data());
    for (int i = 0; i < C3NUM; ++i) {
      RowMajor2Col8MajorFp16(weight_hidden_src + i * hidden_size * weight_shape[1],
                             static_cast<float16_t *>(weight_hidden_) + i * col_align * weight_shape[1], hidden_size,
                             hidden_size, false);
    }
  }
  auto bias_pack_size = static_cast<size_t>(col_align) * sizeof(float16_t);
  auto bias = reinterpret_cast<float16_t *>(malloc(bias_pack_size * C6NUM));
  if (bias == nullptr) {
    MS_LOG(ERROR) << "malloc for packing bias failed.";
    return lite::RET_NULL_PTR;
  }
  (void)memset(bias, 0, bias_pack_size * C6NUM);
  bias_in_ = bias;
  bias_hidden_ = bias + col_align * C3NUM;
  auto bias_in_src = static_cast<const float16_t *>(in_tensors_[FOURTH_INPUT]->data());
  for (int i = 0; i < C3NUM; ++i) {
    (void)memcpy(bias + i * col_align, bias_in_src + i * hidden_size, hidden_size * sizeof(float16_t));
  }
  auto bias_hidden_src = static_cast<const float16_t *>(in_tensors_[FIFTH_INPUT]->data());
  for (int i = 0; i < C3NUM; ++i) {
    (void)memcpy(bias + (C3NUM + i) * col_align, bias_hidden_src + i * hidden_size, hidden_size * sizeof(float16_t));
  }
  if (in_tensors_[SIXTH_INPUT]->IsConst()) {
    init_h_ = malloc(in_tensors_[SIXTH_INPUT]->Size());
    MS_CHECK_TRUE_MSG(init_h_ != nullptr, lite::RET_NULL_PTR, "malloc for init-h failed.");
    (void)memcpy(init_h_, in_tensors_[SIXTH_INPUT]->data(), in_tensors_[SIXTH_INPUT]->Size());
  }
  return RET_OK;
}

int CustomGruFp16CPUKernel::Run() {
  auto input = reinterpret_cast<float16_t *>(in_tensors_[FIRST_INPUT]->data());
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
  auto output = reinterpret_cast<float16_t *>(out_tensors_.front()->data());
  if (output == nullptr) {
    MS_LOG(ERROR) << "Built-in CustomGru's output is nullptr." << name_;
    return lite::RET_NULL_PTR;
  }
  MallocRunBuffer(sizeof(float16_t));
  if (run_buffer_ == nullptr) {
    MS_LOG(ERROR) << "malloc running buffer failed." << name_;
    return lite::RET_NULL_PTR;
  }
  auto param = reinterpret_cast<CustomGruParameter *>(op_parameter_);
  auto row_align = UP_ROUND(param->batch_size, row_tile_);
  auto run_buffer = reinterpret_cast<float16_t *>(run_buffer_);
  float16_t *buffer[C4NUM] = {
    run_buffer, run_buffer + row_align * param->input_size,
    run_buffer + row_align * param->input_size + param->batch_size * param->hidden_size * C3NUM,
    run_buffer + row_align * (param->input_size + param->hidden_size) + param->batch_size * param->hidden_size * C3NUM};
  CustomGruFp16(output, input, static_cast<float16_t *>(weight_in_), static_cast<float16_t *>(weight_hidden_),
                static_cast<float16_t *>(bias_in_), static_cast<float16_t *>(bias_hidden_),
                static_cast<float16_t *>(init_h_), buffer, param);
  if (ms_context_->allocator != nullptr) {
    ms_context_->allocator->Free(run_buffer_);
  } else {
    free(run_buffer_);
  }
  run_buffer_ = nullptr;
  return RET_OK;
}

REG_KERNEL(kCPU, kNumberTypeFloat16, PrimType_Inner_CustomGru, LiteKernelCreator<CustomGruFp16CPUKernel>)
}  // namespace mindspore::kernel
#endif
