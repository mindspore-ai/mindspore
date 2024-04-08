/**
 * Copyright 2021-2023 Huawei Technologies Co., Ltd
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
#include "ms_kernel/random_shuffle.h"
#include <random>
#include <complex>
#include <algorithm>
#include "context/inc/cpu_kernel_utils.h"
#include "random/utils.h"

namespace {
const char *kRandomShuffle = "RandomShuffle";

#define RANDOM_SHUFFLE_COMPUTE_CASE(DTYPE, TYPE) \
  case (DTYPE): {                                \
    RandomShuffleCompute<TYPE>(ctx, output);     \
    break;                                       \
  }
}  // namespace

namespace aicpu {
uint32_t RandomShuffleCpuKernel::Compute(CpuKernelContext &ctx) {
  Tensor *input_0 = ctx.Input(kFirstInputIndex);
  CUST_KERNEL_CHECK_NULLPTR(ctx, input_0, KERNEL_STATUS_PARAM_INVALID, "Get input failed")

  Tensor *output = ctx.Output(kFirstOutputIndex);
  CUST_KERNEL_CHECK_NULLPTR(ctx, output, KERNEL_STATUS_PARAM_INVALID, "Get output failed")
  auto data_type = static_cast<DataType>(output->GetDataType());

  switch (data_type) {
    RANDOM_SHUFFLE_COMPUTE_CASE(DT_FLOAT16, Eigen::half)
    RANDOM_SHUFFLE_COMPUTE_CASE(DT_FLOAT, float)
    RANDOM_SHUFFLE_COMPUTE_CASE(DT_DOUBLE, double)
    RANDOM_SHUFFLE_COMPUTE_CASE(DT_INT8, int8_t)
    RANDOM_SHUFFLE_COMPUTE_CASE(DT_INT16, int16_t)
    RANDOM_SHUFFLE_COMPUTE_CASE(DT_INT32, int32_t)
    RANDOM_SHUFFLE_COMPUTE_CASE(DT_INT64, int64_t)
    RANDOM_SHUFFLE_COMPUTE_CASE(DT_UINT8, uint8_t)
    RANDOM_SHUFFLE_COMPUTE_CASE(DT_UINT16, uint16_t)
    RANDOM_SHUFFLE_COMPUTE_CASE(DT_UINT32, uint32_t)
    RANDOM_SHUFFLE_COMPUTE_CASE(DT_UINT64, uint64_t)
    RANDOM_SHUFFLE_COMPUTE_CASE(DT_BOOL, bool)
    RANDOM_SHUFFLE_COMPUTE_CASE(DT_COMPLEX64, std::complex<float>)
    RANDOM_SHUFFLE_COMPUTE_CASE(DT_COMPLEX128, std::complex<double>)
    default:
      CUST_KERNEL_LOG_ERROR(ctx, "RandomShuffle kernel data type [%s] not support.", DTypeStr(data_type).c_str());
      return KERNEL_STATUS_PARAM_INVALID;
  }
  return KERNEL_STATUS_OK;
}

template <typename T>
uint32_t RandomShuffleCpuKernel::RandomShuffleCompute(CpuKernelContext &ctx, Tensor *output) {
  uint64_t rng_seed = std::random_device()();
  auto rng = std::default_random_engine();
  rng.seed(rng_seed);

  Tensor *input = ctx.Input(kFirstInputIndex);
  auto input_shape = input->GetTensorShape()->GetDimSizes();
  auto input_data = reinterpret_cast<uint8_t *>(input->GetData());
  Tensor *output_ = ctx.Output(kFirstOutputIndex);
  auto output_data = reinterpret_cast<uint8_t *>(output_->GetData());

  if (input_shape.size() <= 1) {
    std::copy(input_data, input_data + input->GetDataSize(), output_data);
    return KERNEL_STATUS_OK;
  }

  size_t batch_size = input_shape[0];
  size_t block_num = std::accumulate(input_shape.begin() + 1, input_shape.end(), 1, std::multiplies<int64_t>());
  size_t block_size = GetSizeByDataType(input->GetDataType()) * block_num;

  std::vector<int64_t> shuffled_indices(batch_size);
  std::iota(shuffled_indices.begin(), shuffled_indices.end(), 0);
  std::shuffle(shuffled_indices.begin(), shuffled_indices.end(), rng);

  auto shuffle_worker = [&shuffled_indices, input_data, output_data, block_size](size_t start, size_t end) {
    for (size_t i = start; i < end; ++i) {
      auto shuffled_idx = shuffled_indices[i];
      auto input_offset = i * block_size;
      auto output_offset = shuffled_idx * block_size;
      std::copy(input_data + input_offset, input_data + input_offset + block_size, output_data + output_offset);
    }
  };

  int core_num = std::max(1, static_cast<int>(aicpu::CpuKernelUtils::GetCPUNum(ctx)) - 2);
  CUST_KERNEL_HANDLE_ERROR(ctx, CpuKernelUtils::ParallelFor(ctx, batch_size, batch_size / core_num, shuffle_worker),
                           "RandomShuffle Compute failed.");
  return KERNEL_STATUS_OK;
}

REGISTER_MS_CPU_KERNEL(kRandomShuffle, RandomShuffleCpuKernel);
}  // namespace aicpu
