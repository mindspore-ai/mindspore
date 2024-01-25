/**
 * Copyright 2024 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/kernel/aicpu/aicpu_ops/poisson_kernels.h"

#include <stdlib.h>
#include <memory.h>
#include <cfloat>
#include <random>

#include "common/random_utils.h"
#include "aicpu_sharder/aicpu_sharder.h"
#include "common/kernel_errcode.h"
#include "common/kernel_log.h"

namespace aicpu {
namespace {
const uint32_t kPoissonShapeIndex = 0;
const uint32_t kPoissonMeanIndex = 1;
const uint32_t kPoissonSeedIndex = 2;
const uint32_t kPoissonSeed2Index = 3;
const uint32_t kPoissonOutputIndex = 4;
static uint64_t PoissonInferOutputShape(std::vector<uint64_t> *out_shape, std::vector<uint64_t> *shape,
                                        std::vector<uint64_t> *mean_shape, uint64_t *count) {
  uint64_t size = (*shape).size() > (*mean_shape).size() ? (*shape).size() : (*mean_shape).size();
  random::NormalizeShape(shape, size);
  random::NormalizeShape(mean_shape, size);
  for (uint64_t i = 0; i < size; ++i) {
    uint64_t shape_n = (*shape)[i] > (*mean_shape)[i] ? (*shape)[i] : (*mean_shape)[i];
    if (((*shape)[i] != 1 && (*shape)[i] != shape_n) || ((*mean_shape)[i] != 1 && (*mean_shape)[i] != shape_n)) {
      return 0;
    }
    (*out_shape).push_back(shape_n);
    (*count) *= shape_n;
  }
  return 1;
}
}  // namespace

uint32_t PoissonKernel::DoCompute() {
  int *tmp_out;
  if (out_count_ > 0) {
    tmp_out = static_cast<int *>(malloc(out_count_ * sizeof(int)));
    if (tmp_out == NULL) {
      return kAicpuKernelStateFailed;
    }
  } else {
    return kAicpuKernelStateFailed;
  }

  uint64_t remainder = out_count_;
  uint64_t mean_remainder = mean_count_;
  std::vector<uint64_t> m_idx(out_count_);
  for (uint64_t i = 0; i < out_shape.size(); ++i) {
    uint64_t mean_pos = 0;
    remainder = remainder / out_shape[i];
    mean_remainder = mean_remainder / mean_shape[i];
    for (uint64_t j = 0; j < out_count_; ++j) {
      m_idx[j] += mean_pos * mean_remainder;
      if ((j + 1) % remainder == 0) {
        ++mean_pos;
        if (mean_pos == mean_shape[i]) {
          mean_pos = 0;
        }
      }
    }
  }

  uint64_t RNG_seed = random::GetRNG(seed_, seed2_);
  std::mt19937 gen(RNG_seed);
  for (uint64_t i = 0; i < out_count_; ++i) {
    std::poisson_distribution<> poisson(mean_[m_idx[i]]);
    tmp_out[i] = poisson(gen);
  }

  int ret = memcpy_s(reinterpret_cast<void *>(io_addrs_[kPoissonOutputIndex]), out_count_ * sizeof(int), tmp_out,
                     out_count_ * sizeof(int));
  free(tmp_out);
  tmp_out = NULL;
  if (ret < 0) {
    return kAicpuKernelStateFailed;
  }
  return kAicpuKernelStateSucess;
}

uint32_t PoissonKernel::ParseKernelParam() {
  seed_ = reinterpret_cast<int64_t *>(io_addrs_[kPoissonSeedIndex]);
  seed2_ = reinterpret_cast<int64_t *>(io_addrs_[kPoissonSeed2Index]);

  // get inputs
  // shape of one batch
  uint64_t tmp_count = 1;
  aicpuops::Tensor shape_tensor = node_def_.inputs(kPoissonShapeIndex);
  aicpuops::TensorShape input_shape = shape_tensor.tensor_shape();
  const auto &shape_dt = static_cast<aicpuops::DataType>(shape_tensor.tensor_type());
  for (int i = 0; i < input_shape.dim_size(); ++i) {
    tmp_count *= input_shape.dim(i).size();
  }
  if (shape_dt == aicpuops::DataType::MS_INT32) {
    auto input0 = reinterpret_cast<int32_t *>(io_addrs_[kPoissonShapeIndex]);
    for (uint64_t index = 0; index < tmp_count; index++) {
      shape.push_back(input0[index]);
    }
  } else {
    auto input0 = reinterpret_cast<int64_t *>(io_addrs_[kPoissonShapeIndex]);
    for (uint64_t index = 0; index < tmp_count; index++) {
      shape.push_back(input0[index]);
    }
  }

  // mean
  mean_ = reinterpret_cast<float *>(io_addrs_[kPoissonMeanIndex]);
  aicpuops::Tensor mean_tensor = node_def_.inputs(kPoissonMeanIndex);
  aicpuops::TensorShape mean_tshape = mean_tensor.tensor_shape();
  for (int i = 0; i < mean_tshape.dim_size(); ++i) {
    mean_shape.push_back(mean_tshape.dim(i).size());
    mean_count_ *= mean_tshape.dim(i).size();
  }

  if (!PoissonInferOutputShape(&out_shape, &shape, &mean_shape, &out_count_)) {
    return kAicpuKernelStateFailed;
  }

  return kAicpuKernelStateSucess;
}
}  // namespace aicpu

extern "C" {
__attribute__((visibility("default"))) uint32_t Poisson(void *param) {
  aicpu::PoissonKernel poissonKernel;
  return poissonKernel.Compute(param);
}
}
