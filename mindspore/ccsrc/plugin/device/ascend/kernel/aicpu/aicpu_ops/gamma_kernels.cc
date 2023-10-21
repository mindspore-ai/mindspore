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
#include "plugin/device/ascend/kernel/aicpu/aicpu_ops/gamma_kernels.h"

#include <stdlib.h>
#include <memory.h>
#include <cfloat>
#include <random>
#include <string>

#include "common/random_utils.h"
#include "aicpu_sharder/aicpu_sharder.h"
#include "common/kernel_errcode.h"
#include "common/kernel_log.h"

namespace aicpu {
namespace {
const uint32_t kGammaShapeIndex = 0;
const uint32_t kGammaAlphaIndex = 1;
const uint32_t kGammaBetaIndex = 2;
const uint32_t kGammaSeedIndex = 3;
const uint32_t kGammaSeed2Index = 4;
const uint32_t kGammaOutputIndex = 5;
}  // namespace
uint32_t GammaKernel::DoCompute() {
  float *tmp_out;
  if (out_count_ > 0) {
    tmp_out = reinterpret_cast<float *>(malloc(out_count_ * sizeof(float)));
    if (tmp_out == NULL) {
      return kAicpuKernelStateFailed;
    }
  } else {
    return kAicpuKernelStateFailed;
  }
  uint64_t remainder = out_count_;
  uint64_t alpha_remainder = alpha_count_;
  uint64_t beta_remainder = beta_count_;
  std::vector<uint64_t> a_idx(out_count_);
  std::vector<uint64_t> b_idx(out_count_);
  for (uint64_t i = 0; i < out_shape.size(); ++i) {
    uint64_t alpha_pos = 0;
    uint64_t beta_pos = 0;
    remainder = remainder / out_shape[i];
    alpha_remainder = alpha_remainder / alpha_shape[i];
    beta_remainder = beta_remainder / beta_shape[i];
    for (uint64_t j = 0; j < out_count_; ++j) {
      a_idx[j] += alpha_pos * alpha_remainder;
      b_idx[j] += beta_pos * beta_remainder;
      if ((j + 1) % remainder == 0) {
        ++alpha_pos;
        ++beta_pos;
        if (alpha_pos == alpha_shape[i]) {
          alpha_pos = 0;
        }
        if (beta_pos == beta_shape[i]) {
          beta_pos = 0;
        }
      }
    }
  }
  // get random generator seed
  uint64_t rng_seed = random::GetRNG(seed_, seed2_);
  std::mt19937 gen(rng_seed);
  for (uint64_t i = 0; i < out_count_; ++i) {
    std::gamma_distribution<float> gamma(alpha_[a_idx[i]], beta_[b_idx[i]]);
    tmp_out[i] = gamma(gen);
  }
  int ret = memcpy_s(reinterpret_cast<void *>(io_addrs_[kGammaOutputIndex]), out_count_ * sizeof(float), tmp_out,
                     out_count_ * sizeof(float));
  free(tmp_out);
  tmp_out = NULL;
  return (ret < 0) ? kAicpuKernelStateFailed : kAicpuKernelStateSucess;
}

uint32_t GammaKernel::ParseKernelParam() {
  // seed
  seed_ = reinterpret_cast<int64_t *>(io_addrs_[kGammaSeedIndex]);
  seed2_ = reinterpret_cast<int64_t *>(io_addrs_[kGammaSeed2Index]);

  // get inputs
  // shape of one batch
  uint64_t tmp_count = 1;
  aicpuops::Tensor shape_tensor = node_def_.inputs(kGammaShapeIndex);
  aicpuops::TensorShape input_shape = shape_tensor.tensor_shape();
  const auto &shape_dt = static_cast<aicpuops::DataType>(shape_tensor.tensor_type());
  for (int i = 0; i < input_shape.dim_size(); ++i) {
    tmp_count *= input_shape.dim(i).size();
  }
  if (shape_dt == aicpuops::DataType::MS_INT32) {
    auto input0 = reinterpret_cast<int32_t *>(io_addrs_[kGammaShapeIndex]);
    for (uint64_t index = 0; index < tmp_count; index++) {
      shape.push_back(input0[index]);
    }
  } else {
    auto input0 = reinterpret_cast<int64_t *>(io_addrs_[kGammaShapeIndex]);
    for (uint64_t index = 0; index < tmp_count; index++) {
      shape.push_back(input0[index]);
    }
  }

  // alpha
  alpha_ = reinterpret_cast<float *>(io_addrs_[kGammaAlphaIndex]);
  aicpuops::Tensor alpha_tensor = node_def_.inputs(kGammaAlphaIndex);
  aicpuops::TensorShape alpha_tshape = alpha_tensor.tensor_shape();
  for (int i = 0; i < alpha_tshape.dim_size(); ++i) {
    alpha_shape.push_back(alpha_tshape.dim(i).size());
    alpha_count_ *= alpha_tshape.dim(i).size();
  }

  // beta
  beta_ = reinterpret_cast<float *>(io_addrs_[kGammaBetaIndex]);
  aicpuops::Tensor beta_tensor = node_def_.inputs(kGammaBetaIndex);
  aicpuops::TensorShape beta_tshape = beta_tensor.tensor_shape();
  for (int i = 0; i < beta_tshape.dim_size(); ++i) {
    beta_shape.push_back(beta_tshape.dim(i).size());
    beta_count_ *= beta_tshape.dim(i).size();
  }

  if (!random::InferOutputShape(&out_shape, &shape, &alpha_shape, &beta_shape, &out_count_)) {
    return kAicpuKernelStateFailed;
  }

  return kAicpuKernelStateSucess;
}
}  // namespace aicpu

extern "C" {
__attribute__((visibility("default"))) uint32_t Gamma(void *param) {
  aicpu::GammaKernel gammaKernel;
  return gammaKernel.Compute(param);
}
}
