/**
 * Copyright 2022-2023 Huawei Technologies Co., Ltd
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
#ifndef AICPU_AICPU_OPS_RANDOM_SHUFFLE_KERNEL_H_
#define AICPU_AICPU_OPS_RANDOM_SHUFFLE_KERNEL_H_

#include <vector>
#include <random>
#include "common/kernel_base.h"

namespace aicpu {
class RandomShuffleKernel : public KernelBase {
 public:
  RandomShuffleKernel()
      : KernelBase("RandomShuffleKernel"), block_num_(1), block_size_(1), data_size_(0), seed_(0), seed2_(0) {}
  ~RandomShuffleKernel() = default;

 protected:
  uint32_t ParseKernelParam() override;
  uint32_t DoCompute() override;

  template <typename Scalar>
  void IndexShuffle(const size_t &size, void *data);
  template <typename Scalar>
  uint32_t ScalarShuffle();
  uint32_t TensorShuffle();

  size_t block_num_;
  size_t block_size_;
  size_t data_size_;
  uint64_t seed_;
  uint64_t seed2_;
  std::default_random_engine rng_;
};
}  // namespace aicpu
#endif  // AICPU_AICPU_OPS_RANDOM_SHUFFLE_KERNEL_H_
