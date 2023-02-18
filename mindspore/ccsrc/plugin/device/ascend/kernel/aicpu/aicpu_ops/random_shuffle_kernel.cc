/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include "plugin/device/ascend/kernel/aicpu/aicpu_ops/random_shuffle_kernel.h"
#include <vector>
#include <string>
#include "securec/include/securec.h"
#include "proto/aicpu_tensor.pb.h"
#include "google/protobuf/stubs/common.h"
#include "./kernel_base.h"
#include "./kernel_errcode.h"
#include "./kernel_log.h"

namespace aicpu {
template <typename Scalar>
void RandomShuffleKernel::IndexShuffle(const size_t &size, void *data) {
  Scalar *perm = reinterpret_cast<Scalar *>(data);
  for (size_t i = size - 1; i > 0; --i) {
    std::uniform_int_distribution<size_t> dist(0, i);
    size_t rng = dist(generator_);
    Scalar tmp = perm[i];
    perm[i] = perm[rng];
    perm[rng] = tmp;
  }
}

template <typename Scalar>
uint32_t RandomShuffleKernel::ScalarShuffle() {
  // Copy input to output, then shuffle output.
  const size_t &input_size = block_num_ * block_size_ * sizeof(Scalar);
  auto ret =
    memcpy_s(reinterpret_cast<void *>(io_addrs_[1]), input_size, reinterpret_cast<void *>(io_addrs_[0]), input_size);
  if (ret != EOK) {
    AICPU_LOGE("memcpy_s() failed: %d.", ret);
    return kAicpuKernelStateInternalError;
  }

  IndexShuffle<Scalar>(block_num_, reinterpret_cast<void *>(io_addrs_[1]));
  return kAicpuKernelStateSucess;
}

uint32_t RandomShuffleKernel::TensorShuffle() {
  std::vector<size_t> permutation(block_num_);
  for (size_t i = 0; i < block_num_; i++) {
    permutation[i] = i;
  }

  IndexShuffle<size_t>(block_num_, permutation.data());

  const size_t &size = block_size_ * data_size_;
  for (size_t i = 0; i < block_num_; i++) {
    auto output_offset = reinterpret_cast<uint8_t *>(io_addrs_[1]) + i * size;
    auto input_offset = reinterpret_cast<uint8_t *>(io_addrs_[0]) + permutation[i] * size;
    auto ret = memcpy_s(reinterpret_cast<void *>(output_offset), size, reinterpret_cast<void *>(input_offset), size);
    if (ret != EOK) {
      AICPU_LOGE("memcpy_s() failed: %d.", ret);
      return kAicpuKernelStateInternalError;
    }
  }

  return kAicpuKernelStateSucess;
}

uint32_t RandomShuffleKernel::ParseKernelParam() {
  ::google::protobuf::Map<::std::string, ::aicpuops::AttrValue> attrs = node_def_.attrs();
  int64_t seed1 = attrs["seed"].i();
  int64_t seed2 = attrs["seed2"].i();
  std::random_device rd;
  int64_t seed = (seed2 != 0) ? seed2 : (seed1 != 0) ? seed1 : rd();
  generator_.seed(static_cast<uint64_t>(seed));

  const size_t &num_input = node_def_.inputs_size();
  if (num_input != 1) {
    AICPU_LOGE("For RandomShuffle: input num should be 1.");
    return kAicpuKernelStateInvalid;
  }

  ::aicpuops::Tensor input = node_def_.inputs(static_cast<int>(0));
  ::aicpuops::TensorShape shape = input.tensor_shape();

  const auto &dtype = static_cast<::aicpuops::DataType>(input.tensor_type());
  data_size_ = GetDataTypeSize(dtype);

  size_t dim = IntToSize(shape.dim_size());
  if (dim == 0) {
    // Input is a scalar: keep block number to be 1.
    return kAicpuKernelStateSucess;
  }
  block_num_ = static_cast<size_t>(shape.dim(0).size());

  for (size_t i = 1; i < dim; i++) {
    block_size_ *= static_cast<size_t>(shape.dim(SizeToInt(i)).size());
  }

  return kAicpuKernelStateSucess;
}

uint32_t RandomShuffleKernel::DoCompute() {
  if (block_num_ == 1) {
    const size_t size_in_bytes = block_size_ * data_size_;
    auto ret = memcpy_s(reinterpret_cast<void *>(io_addrs_[1]), size_in_bytes, reinterpret_cast<void *>(io_addrs_[0]),
                        size_in_bytes);
    if (ret != EOK) {
      AICPU_LOGE("memcpy_s() failed: %d.", ret);
      return kAicpuKernelStateInternalError;
    }
  }

  const size_t &block_size_in_bytes = block_size_ * data_size_;
  switch (block_size_in_bytes) {
    case sizeof(int8_t):
      return ScalarShuffle<int8_t>();
    case sizeof(int16_t):
      return ScalarShuffle<int16_t>();
    case sizeof(int32_t):
      return ScalarShuffle<int32_t>();
    case sizeof(int64_t):
      return ScalarShuffle<int64_t>();
    default:
      return TensorShuffle();
  }
}
}  // namespace aicpu

extern "C" {
__attribute__((visibility("default"))) uint32_t RandomShuffle(void *param) {
  aicpu::RandomShuffleKernel random_shuffle_kernel;
  return random_shuffle_kernel.Compute(param);
}
}
